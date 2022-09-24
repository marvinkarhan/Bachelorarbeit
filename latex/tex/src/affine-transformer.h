void affineTransform(const std::uint8_t *input, std::int32_t *output)
{
  const auto inputVector = reinterpret_cast<const __m256i *>(input);
  const __m256i ones = _mm256_set1_epi16(1);
  for (std::uint32_t i = 0; i < ouputDimensions; ++i)
  {
    const std::uint32_t offset = i * inputDimensions;
    __m256i sum = _mm256_setzero_si256();
    const auto row = reinterpret_cast<const __m256i *>(&weights_[offset]);

    constexpr std::uint32_t kNumChunks = inputDimensions / 32;
    for (std::uint32_t j = 0; j < kNumChunks; ++j)
    {
      __m256i product = _mm256_maddubs_epi16(&inputVector[j], &row[j]);
      product = _mm256_madd_epi16(product, ones);
      sum = _mm256_add_epi32(sum, product);
    }

    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
    output[i] = _mm_cvtsi128_si32(sum128) + biases_[i];
  }
}