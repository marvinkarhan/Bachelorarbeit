alignas(64) std::int16_t accumulator[2][256];

class Board;
class IndexList;

void refreshAccumulator(Board &board)
{
  std::int16_t(&accumulator)[2][256] = board.state->accumulator;
  for (bool perspective : {WHITE, BLACK})
  {
    IndexList activeIndices;
    AppendActiveIndices(board, perspective, activeIndices);
    std::memcpy(accumulator[perspective], biases, 256 * std::int16_t);
    for (const auto index : activeIndices)
    {
      const std::uint32_t offset = 256 * index;
      auto accumulation = reinterpret_cast<__m256i *>(&accumulator[perspective]);
      auto column = reinterpret_cast<const __m256i *>(&weights[offset]);
      constexpr std::uint32_t numChunks = 256 / 16;
      for (std::uint32_t j = 0; j < numChunks; ++j)
      {
        __m256i acc = _mm256_load_si256(&accumulation[j]);
        __m256i sum = _mm256_add_epi16(acc, column[j]);
        _mm256_store_si256(&accumulation[j], sum);
      }
    }
  }
}

void updateAccumulator(Board &board)
{
  std::int16_t(&accumulator)[2][256] = board.state->accumulator;
  for (bool perspective : {WHITE, BLACK})
  {
    IndexList removed_indices, added_indices;
    AppendChangedIndices(board, perspective, removed_indices, added_indices);
    constexpr std::uint32_t numChunks = 256 / 16;
    auto accumulation = reinterpret_cast<__m256i *>(&accumulator[perspective]);
    // Difference calculation for the deactivated features
    for (const auto index : removed_indices)
    {
      const std::uint32_t offset = 256 * index;
      auto column = reinterpret_cast<const __m256i *>(&weights[offset]);
      for (std::uint32_t j = 0; j < numChunks; ++j)
        accumulation[j] = _mm256_sub_epi16(accumulation[j], column[j]);
    }
    // Difference calculation for the activated features
    for (const auto index : added_indices)
    {
      auto column = reinterpret_cast<const __m256i *>(&weights[offset]);
      for (std::uint32_t j = 0; j < numChunks; ++j)
        accumulation[j] = _mm256_add_epi16(accumulation[j], column[j]);
    }
  }
}

void clippedReLU(Board &board, std::uint8_t *output)
{
  std::int16_t(&accumulator)[2][256] = board.state->accumulator;
  auto accumulation = reinterpret_cast<__m256i *>(&accumulator[perspective]);
  constexpr std::uint32_t numChunks = 256 / 32;
  constexpr int kControl = 0b11011000;
  const __m256i kZero = _mm256_setzero_si256();
  const bool perspectives[2] = {board.activeSide, !board.activeSide};
  for (std::uint32_t p = 0; p < 2; ++p)
  {
    const std::uint32_t offset = kHalfDimensions * p;
    auto out = reinterpret_cast<__m256i *>(&output[offset]);
    for (std::uint32_t j = 0; j < numChunks; ++j)
    {
      __m256i sum0 = _mm256_load_si256(&reinterpret_cast<const __m256i *>(accumulation[perspectives[p]])[j * 2 + 0]);
      __m256i sum1 = _mm256_load_si256(&reinterpret_cast<const __m256i *>(accumulation[perspectives[p]])[j * 2 + 1]);
      _mm256_store_si256(
          &out[j],
          _mm256_permute4x64_epi64(
              _mm256_max_epi8(
                  _mm256_packs_epi16(sum0, sum1),
                  kZero),
              kControl));
    }
  }
}