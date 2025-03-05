## Which one should be used?

| Scenario                                          | Cosine Similarity          | Euclidean Distance                |
| ------------------------------------------------- | -------------------------- | --------------------------------- |
| High-dimensional embeddings (e.g., SBERT, Gemini) | ✅ Best choice              | ❌ Not effective                   |
| Low-dimensional numerical data                    | ❌ Less effective           | ✅ Works well                      |
| Recommendation systems                            | ✅ Common choice            | ⚠️ Can work if dimensions are low |
| Clustering (e.g., k-means)                        | ✅ If using text embeddings | ✅ If using numeric data           |
**Cosine Similarity is usually better when:**
- You have high dimensional spaces (e.g. 384+ dimensions).
- Your text embeddings are normalised.
- You want to measure the angle instead of the distance.

**Euclidean Distance is usually better when:**
- You have low dimensional spaces (e.g. 2D or 3D).
- The magnitude of difference matters (this gets normalised in Cosine Similarity calculations).
- You have dense, structured data.

Benchmark results of using Cosine Similarity can be found here: [[Embedding Benchmarks]]
