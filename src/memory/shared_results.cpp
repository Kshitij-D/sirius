#include "memory/shared_results.hpp"

#include "memory/resident_ledger.hpp"

#include <cucascade/data/data_batch.hpp>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace sirius::memory {

namespace {

struct registry {
  std::mutex mu;
  std::unordered_set<std::uint64_t> allowed;
  std::unordered_map<std::uint64_t, std::shared_ptr<const shared_result>> published;
  std::unordered_set<std::uint64_t> refused;  // over budget: logged once
  std::size_t used = 0;
  std::size_t budget = std::size_t{4} << 30;
  bool on = false;
};

void load(registry& x)
{
  auto const* path = std::getenv("SIRIUS_SHARE_KEYS");
  if (path == nullptr || *path == '\0') { return; }
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "[share] cannot read SIRIUS_SHARE_KEYS=%s; sharing off\n", path);
    return;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) { x.allowed.insert(std::stoull(line, nullptr, 16)); }
  }
  if (auto const* b = std::getenv("SIRIUS_SHARE_BUDGET_BYTES"); b != nullptr && *b != '\0') {
    x.budget = std::stoull(b);
  }
  x.on = true;
  std::fprintf(stderr, "[share] on: %zu keys, budget %zu bytes\n", x.allowed.size(), x.budget);
}

registry& reg()
{
  static registry r;
  static std::once_flag once;
  std::call_once(once, [] { load(r); });
  return r;
}

bool is_word(char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_'; }
bool is_space(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }

std::size_t skip_space(const std::string& s, std::size_t i)
{
  while (i < s.size() && is_space(s[i])) { ++i; }
  return i;
}

bool match_word_ci(const std::string& s, std::size_t i, const char* w)
{
  for (std::size_t k = 0; w[k] != '\0'; ++k) {
    if (i + k >= s.size() ||
        std::tolower(static_cast<unsigned char>(s[i + k])) != w[k]) {
      return false;
    }
  }
  return true;
}

struct cte_text {
  std::string name;
  std::string body;
};

// Same grammar as the batch planner (exp60/merge2.py ctes()): WITH name [(cols)] AS ( body ) [, ...]
std::vector<cte_text> parse_ctes(const std::string& sql)
{
  std::vector<cte_text> out;
  auto i = skip_space(sql, 0);
  if (!match_word_ci(sql, i, "with") || i + 4 >= sql.size() || !is_space(sql[i + 4])) { return out; }
  i = skip_space(sql, i + 4);
  while (true) {
    auto const n0 = i;
    while (i < sql.size() && is_word(sql[i])) { ++i; }
    if (i == n0) { return out; }
    cte_text c{sql.substr(n0, i - n0), {}};
    auto j = i;
    if (auto const k = skip_space(sql, i); k < sql.size() && sql[k] == '(') {
      j = k;  // column list without nested parentheses
      while (j < sql.size() && sql[j] != ')') { ++j; }
      if (j >= sql.size()) { return out; }
      ++j;
    }
    auto const before_as = j;
    j                    = skip_space(sql, j);
    if (j == before_as || !match_word_ci(sql, j, "as")) { return out; }
    j = skip_space(sql, j + 2);
    if (j >= sql.size() || sql[j] != '(') { return out; }
    auto const b0 = ++j;
    int depth     = 1;
    while (j < sql.size() && depth > 0) {
      if (sql[j] == '(') { ++depth; }
      if (sql[j] == ')') { --depth; }
      ++j;
    }
    if (depth != 0) { return out; }
    c.body = sql.substr(b0, j - 1 - b0);
    out.push_back(std::move(c));
    j = skip_space(sql, j);
    if (j >= sql.size() || sql[j] != ',') { return out; }
    i = skip_space(sql, j + 1);
  }
}

std::string collapse_space(const std::string& s)
{
  std::string out;
  bool pending = false;
  for (char c : s) {
    if (is_space(c)) {
      pending = true;
      continue;
    }
    if (pending && !out.empty()) { out.push_back(' '); }
    pending = false;
    out.push_back(c);
  }
  return out;
}

bool references(const std::string& body, const std::string& name)
{
  for (auto p = body.find(name); p != std::string::npos; p = body.find(name, p + 1)) {
    bool const left  = p == 0 || !is_word(body[p - 1]);
    bool const right = p + name.size() >= body.size() || !is_word(body[p + name.size()]);
    if (left && right) { return true; }
  }
  return false;
}

std::uint64_t fnv1a(const std::string& s)
{
  std::uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}

}  // namespace

bool sharing_enabled() noexcept { return reg().on; }

std::optional<std::uint64_t> cte_share_key(const std::string& sql, const std::string& ctename)
{
  auto const ctes = parse_ctes(sql);
  std::vector<std::uint64_t> keys;
  for (std::size_t i = 0; i < ctes.size(); ++i) {
    std::string text = collapse_space(ctes[i].body) + "|";
    bool first       = true;
    for (std::size_t j = 0; j < i; ++j) {
      if (!references(ctes[i].body, ctes[j].name)) { continue; }
      char hex[17];
      std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(keys[j]));
      text += (first ? "" : ",") + std::string(hex);
      first = false;
    }
    keys.push_back(fnv1a(text));
    if (ctes[i].name == ctename) { return keys.back(); }
  }
  return std::nullopt;
}

bool share_key_allowed(std::uint64_t key) noexcept
{
  auto& r = reg();
  return r.on && r.allowed.count(key) != 0;
}

std::shared_ptr<const shared_result> find_shared_result(std::uint64_t key)
{
  auto& r = reg();
  std::lock_guard lk(r.mu);
  auto const it = r.published.find(key);
  return it == r.published.end() ? nullptr : it->second;
}

void publish_shared_result(std::uint64_t key,
                           std::vector<std::shared_ptr<::cucascade::data_batch>> batches,
                           duckdb::vector<duckdb::LogicalType> logical_types)
{
  auto& r = reg();
  std::size_t bytes                              = 0;
  const ::cucascade::memory::memory_space* space = nullptr;
  for (auto const& b : batches) {
    auto ro = b->to_read_only();
    bytes += ro.get_data()->get_size_in_bytes();
    if (space == nullptr) { space = ro.get_memory_space(); }
  }
  std::lock_guard lk(r.mu);
  if (r.published.count(key) != 0) { return; }
  if (r.used + bytes > r.budget) {
    if (r.refused.insert(key).second) {
      std::fprintf(stderr, "[share] over budget key=%016llx bytes=%zu used=%zu\n",
                   static_cast<unsigned long long>(key), bytes, r.used);
    }
    return;
  }
  auto e           = std::make_shared<shared_result>();
  e->batches       = std::move(batches);
  e->logical_types = std::move(logical_types);
  e->bytes         = bytes;
  r.used += bytes;
  add_resident(space, bytes);  // held until the process ends: the reservation cap must see it
  std::fprintf(stderr, "[share] publish key=%016llx batches=%zu bytes=%zu used=%zu\n",
               static_cast<unsigned long long>(key), e->batches.size(), bytes, r.used);
  r.published.emplace(key, std::move(e));
}

void note_shared_hit(std::uint64_t key, std::size_t bytes)
{
  std::fprintf(stderr, "[share] hit key=%016llx bytes=%zu\n", static_cast<unsigned long long>(key),
               bytes);
}

}  // namespace sirius::memory
