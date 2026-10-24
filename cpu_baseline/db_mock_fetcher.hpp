// cpu_baseline/db_mock_fetcher.hpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// Multi-tenant key store: one ML-DSA secret key per tenant, held in host memory.
struct TenantKeyStore {
    size_t num_tenants = 0;
    size_t sk_len = 0;
    size_t pk_len = 0;
    std::vector<uint8_t> sks;  // num_tenants * sk_len, tenant-major
    std::vector<uint8_t> pks;  // num_tenants * pk_len, tenant-major

    const uint8_t* sk(size_t tenant) const { return sks.data() + tenant * sk_len; }
    const uint8_t* pk(size_t tenant) const { return pks.data() + tenant * pk_len; }
};