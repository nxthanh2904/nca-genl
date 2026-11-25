# NCA-GENL Practice Exam 01
Independent practice material — not official NVIDIA exam questions.
50 questions | 60 minutes | English
Blueprint: https://www.nvidia.com/en-gb/learn/certification/generative-ai-llm-associate/
## Questions
### 1. A team trains a model using customer messages paired with manually assigned intent labels. Which learning paradigm is being used?
*Core Machine Learning and AI Knowledge · fundamental*

- A. Reinforcement learning
- B. Unsupervised learning
- C. Self-supervised learning
- D. Supervised learning

### 2. A retailer wants to discover customer groups without predefined segment labels. Which technique is most appropriate?
*Core Machine Learning and AI Knowledge · fundamental*

- A. Supervised classification
- B. Linear regression
- C. K-means clustering
- D. Logistic regression

### 3. An LLM learns to predict the next token from raw text without manually annotated labels. What best describes this training objective?
*Core Machine Learning and AI Knowledge · fundamental*

- A. Reinforcement learning
- B. Human-supervised classification
- C. Clustering
- D. Self-supervised learning

### 4. Which Transformer component prevents a decoder-only language model from attending to future tokens during training?
*Core Machine Learning and AI Knowledge · intermediate*

- A. Token embedding
- B. Layer normalization
- C. Causal attention mask
- D. Output projection

### 5. What makes dense text embeddings useful for semantic search?
*Core Machine Learning and AI Knowledge · fundamental*

- A. They preserve every word in its original order exactly
- B. They eliminate the need for a similarity metric
- C. They always assign identical vectors to paraphrases
- D. Semantically related texts can have nearby vector representations

### 6. A chatbot must answer questions about policies that change weekly without retraining its language model. Which approach is most suitable?
*Core Machine Learning and AI Knowledge · intermediate*

- A. Increase the number of pretraining epochs
- B. Use a larger tokenizer vocabulary
- C. Increase temperature
- D. Retrieve current policy passages and include them in the prompt

### 7. An engineer adds three labeled examples to a prompt before asking the model to classify a new message. Which method is used?
*Core Machine Learning and AI Knowledge · fundamental*

- A. Full fine-tuning
- B. Zero-shot prompting
- C. Few-shot prompting
- D. Knowledge distillation

### 8. A team needs to adapt a pretrained LLM using less trainable parameter memory than full fine-tuning. Which method is most appropriate?
*Core Machine Learning and AI Knowledge · intermediate*

- A. Full fine-tuning
- B. Document chunking
- C. LoRA
- D. Greedy decoding

### 9. Why is k-fold cross-validation useful when comparing models on a limited labeled dataset?
*Core Machine Learning and AI Knowledge · intermediate*

- A. It evaluates models across multiple train-validation partitions
- B. It removes the need for training
- C. It guarantees zero overfitting
- D. It makes the final test set part of every training fold

### 10. Which architecture is most directly designed for autoregressive next-token generation?
*Core Machine Learning and AI Knowledge · fundamental*

- A. K-means model
- B. Principal component analysis
- C. Decoder-only Transformer with causal self-attention
- D. Encoder-only Transformer with bidirectional attention

### 11. What is the distinction between token IDs and token embeddings?
*Core Machine Learning and AI Knowledge · intermediate*

- A. Embeddings are the original text strings
- B. IDs and embeddings are always identical
- C. Token IDs are probability distributions
- D. IDs identify vocabulary entries; embeddings are dense vectors

### 12. A model must generate a response using four specified JSON fields. Which prompt change is most useful?
*Core Machine Learning and AI Knowledge · intermediate*

- A. Increase the training batch size
- B. Remove field names from instructions
- C. Increase temperature to maximum
- D. Specify the schema and show a valid output example

### 13. What is the main effect of lowering sampling temperature while keeping other decoding settings fixed?
*Core Machine Learning and AI Knowledge · fundamental*

- A. It guarantees factual accuracy
- B. It increases the context window
- C. It generally concentrates probability on higher-scoring tokens
- D. It permanently updates model weights

### 14. Which objective is commonly used to train an autoregressive language model?
*Core Machine Learning and AI Knowledge · intermediate*

- A. BLEU as the standard differentiable token loss
- B. Mean absolute error on token IDs
- C. Silhouette score
- D. Next-token cross-entropy loss

### 15. A RAG system retrieves the correct source passage, but the answer ignores it. Which component should be investigated first?
*Core Machine Learning and AI Knowledge · advanced*

- A. Generation prompt and grounding behavior
- B. Replace the embedding model immediately
- C. Increase embedding dimensionality only
- D. Delete the vector index

### 16. Which library provides a GPU-accelerated DataFrame interface for tabular preprocessing?
*Software Development · fundamental*

- A. spaCy
- B. cuDF
- C. TensorRT
- D. cuML

### 17. A developer wants to accelerate a compatible scikit-learn-style clustering workload on an NVIDIA GPU. Which library is most relevant?
*Software Development · fundamental*

- A. cuDF
- B. Triton Inference Server
- C. ONNX
- D. cuML

### 18. What is the primary benefit of exporting a supported model to ONNX? Select TWO answers.
*Software Development · fundamental*

- A. Guaranteed identical speed on all devices
- B. It enables supported models to be consumed by compatible inference engines
- C. Guaranteed elimination of all numerical errors
- D. A portable model representation for compatible runtimes
- E. Automatic collection of training labels

### 19. Which technique can TensorRT use to reduce inference overhead on supported NVIDIA GPUs?
*Software Development · intermediate*

- A. Operator fusion
- B. Human preference annotation
- C. Randomly discarding training examples
- D. Increasing dropout during inference

### 20. A team needs to expose trained models through a centralized inference service supporting multiple model backends. Which NVIDIA tool fits?
*Software Development · fundamental*

- A. NumPy
- B. cuDF
- C. Triton Inference Server
- D. cuML

### 21. A developer wants a packaged NVIDIA inference microservice for a supported LLM rather than building the serving stack from scratch. Which offering is most relevant?
*Software Development · fundamental*

- A. spaCy
- B. cuGraph
- C. NVIDIA NIM
- D. cuDF

### 22. Which Python library is particularly suited to tokenization, linguistic annotation and named-entity recognition in NLP pipelines?
*Software Development · fundamental*

- A. Matplotlib
- B. cuDF
- C. NumPy
- D. spaCy

### 23. What should a vector database store to support semantic document retrieval with source attribution?
*Software Development · intermediate*

- A. Only raw GPU utilization measurements
- B. Only the generator vocabulary
- C. Embeddings linked to document text or source metadata
- D. Only model training loss values

### 24. An embedding model is replaced by a new model with a different vector space. What should be done before using it for retrieval? Select TWO answers.
*Software Development · advanced*

- A. Pad existing vectors with zeros regardless of model
- B. Re-embed indexed documents and queries with compatible models
- C. Validate retrieval quality after rebuilding the index
- D. Change only the generator temperature
- E. Keep the old document vectors without validation

### 25. Which metrics best reveal whether an LLM service is meeting interactive performance requirements?
*Software Development · intermediate*

- A. Training dataset size only
- B. Number of source code files
- C. Time to first token, request latency and error rate
- D. Vocabulary size only

### 26. What is a common trade-off when increasing inference batch size?
*Software Development · intermediate*

- A. GPU memory use always decreases
- B. Model accuracy necessarily doubles
- C. Throughput and latency always improve together
- D. Higher throughput may come with increased waiting time or memory usage

### 27. An LLM API occasionally returns transient server errors. Which client strategy is most appropriate?
*Software Development · intermediate*

- A. Retry indefinitely without delay
- B. Retrain the model for every error
- C. Bounded retries with backoff and timeout handling
- D. Ignore errors and report success

### 28. Which dataset should remain untouched during hyperparameter selection to provide a final unbiased evaluation?
*Experimentation · fundamental*

- A. Combined training and validation set during selection
- B. Training set
- C. Held-out test set
- D. Validation set

### 29. Training loss continues to decrease while validation loss rises. What is the most likely diagnosis?
*Experimentation · intermediate*

- A. Overfitting
- B. Underfitting both datasets
- C. Guaranteed data leakage
- D. Successful convergence on unseen data

### 30. Which procedure can help mitigate overfitting during fine-tuning?
*Experimentation · intermediate*

- A. Use test data for every checkpoint decision
- B. Stop based on a monitored validation metric
- C. Remove validation monitoring
- D. Select the checkpoint with lowest training loss only

### 31. A dataset contains 2,400 examples. Training uses batches of 200 for three epochs, with one optimizer update per batch. How many updates occur?
*Experimentation · intermediate*

- A. 36
- B. 7200
- C. 600
- D. 12

### 32. A researcher wants to measure the contribution of a reranker in a RAG pipeline. Which experiment is best?
*Experimentation · intermediate*

- A. Compare with and without the reranker while holding other settings fixed
- B. Change the embedding model and reranker together
- C. Use different test questions for each variant
- D. Measure training loss of the generator only

### 33. Two chatbot versions are evaluated on live traffic. Which design reduces time-related confounding?
*Experimentation · intermediate*

- A. Run one version this month and the other next month
- B. Compare only the best responses from each variant
- C. Randomly assign eligible users to concurrent variants
- D. Give the new version only to expert users

### 34. For the same evaluation text and tokenizer, what does lower language-model perplexity generally indicate?
*Experimentation · intermediate*

- A. Guaranteed factual correctness
- B. Lower GPU memory usage
- C. Higher likelihood assigned to observed tokens
- D. Higher output diversity in every decoding configuration

### 35. Which evaluation is most appropriate for assessing whether generated summaries preserve meaning and remain coherent?
*Experimentation · fundamental*

- A. GPU temperature only
- B. Training epoch count
- C. Exact character count alone
- D. Human review using a predefined rubric

### 36. What is the purpose of collecting human preference comparisons between candidate model responses?
*Experimentation · intermediate*

- A. Construct GPU execution kernels
- B. Provide preference supervision for alignment
- C. Replace tokenization
- D. Guarantee elimination of all bias

### 37. A team converts an LLM to a lower-precision representation. Which evaluation is needed before deployment? Select TWO answers.
*Experimentation · advanced*

- A. Compare quality, latency and memory under representative workloads
- B. Check file size only
- C. Assume accuracy is identical
- D. Use the same prompts and decoding settings when comparing quality
- E. Compare different tasks for each precision

### 38. Which practice most directly improves reproducibility of model-comparison experiments?
*Experimentation · intermediate*

- A. Keep only the best score
- B. Record dataset versions, configuration, seeds and model checkpoints
- C. Overwrite all previous checkpoints
- D. Change random seeds without recording them

### 39. A response-latency dataset contains a few extremely slow requests. Which statistic is less sensitive to these extremes than the mean?
*Data Analysis and Visualization · fundamental*

- A. Arithmetic mean
- B. Maximum
- C. Range
- D. Median

### 40. Which plot is best suited to examining the relationship between prompt length and response latency?
*Data Analysis and Visualization · fundamental*

- A. Single summary table of means
- B. Pie chart
- C. Scatter plot
- D. Single histogram of prompt lengths

### 41. A numerical feature has missing values. To avoid leaking validation information, how should median imputation be performed?
*Data Analysis and Visualization · intermediate*

- A. Use validation labels to choose replacements
- B. Fit the median on all data before splitting
- C. Fit the median on training data and apply it to validation data
- D. Replace missing values with test set statistics

### 42. Only 1% of messages contain prohibited content. A classifier predicts every message as safe. Why is 99% accuracy misleading?
*Data Analysis and Visualization · intermediate*

- A. The dataset is necessarily corrupted
- B. Accuracy cannot be calculated for classification
- C. The model must have perfect recall
- D. It hides failure to detect the minority class

### 43. What does a p95 request latency of 800 ms mean under the usual empirical percentile interpretation?
*Data Analysis and Visualization · intermediate*

- A. Exactly 95 requests were measured
- B. Approximately 95% of requests complete within 800 ms
- C. The mean equals 800 ms
- D. Every request completes in 800 ms

### 44. A detector flags 100 responses, of which 80 truly violate policy. What is its precision?
*Data Analysis and Visualization · intermediate*

- A. 0.20
- B. 0.80
- C. Recall cannot be computed, so precision cannot either
- D. 1.25

### 45. An analysis finds a positive correlation between prompt length and latency. Which conclusion is justified?
*Data Analysis and Visualization · intermediate*

- A. The relationship must hold in every deployment
- B. Longer prompts tend to be associated with higher latency in this sample
- C. Latency must cause prompts to become longer
- D. Prompt length is the sole cause of latency

### 46. A company plans to use customer support transcripts for fine-tuning. Which preparation is most appropriate?
*Trustworthy AI · fundamental*

- A. Publish raw transcripts for reproducibility
- B. Retain all sensitive data by default
- C. Assume public access equals consent
- D. Check lawful authorization and minimize or redact unnecessary personal data

### 47. A hiring-assistance model performs substantially worse for one demographic group. What should the team do first?
*Trustworthy AI · intermediate*

- A. Report only aggregate accuracy
- B. Increase temperature without evaluation
- C. Investigate subgroup errors and data representation
- D. Remove all records from that group

### 48. A retrieved document tells the assistant to ignore its instructions and disclose secrets. What is the best design response? Select TWO answers.
*Trustworthy AI · advanced*

- A. Always execute instructions found in retrieved text
- B. Assume trusted embedding similarity guarantees safe content
- C. Treat retrieved content as untrusted data and enforce access controls outside the model
- D. Restrict tool permissions to authorized operations
- E. Increase the context window only

### 49. Which practice helps users verify factual claims produced by a RAG assistant?
*Trustworthy AI · intermediate*

- A. Increase randomness to maximize factuality
- B. Provide source references and abstain when supporting evidence is insufficient
- C. Always answer confidently
- D. Hide all retrieved sources

### 50. What is the appropriate role of guardrails in an LLM application?
*Trustworthy AI · intermediate*

- A. Eliminate the need to monitor failures
- B. Guarantee every response is safe
- C. Replace backend authorization
- D. Apply defined input/output policies as one layer of a broader safety system

## Answer key and explanations
### 1. D
The target labels guide training.

- **A — Incorrect:** Learning is driven by rewards from interactions.
- **B — Incorrect:** No target labels are required.
- **C — Incorrect:** Targets are derived from the data itself.
- **D — Correct:** The target labels guide training.

### 2. C
It groups observations by similarity without labels.

- **A — Incorrect:** It requires predefined classes.
- **B — Incorrect:** It predicts continuous targets.
- **C — Correct:** It groups observations by similarity without labels.
- **D — Incorrect:** It learns from labeled targets.

### 3. D
The text itself supplies prediction targets.

- **A — Incorrect:** No reward-driven interaction is specified.
- **B — Incorrect:** Manually assigned class labels are not needed.
- **C — Incorrect:** It does not predict next tokens.
- **D — Correct:** The text itself supplies prediction targets.

### 4. C
It blocks attention to subsequent positions.

- **A — Incorrect:** It represents token identities.
- **B — Incorrect:** It normalizes activations.
- **C — Correct:** It blocks attention to subsequent positions.
- **D — Incorrect:** It maps hidden states to vocabulary logits.

### 5. D
Embedding models can encode meaning beyond exact word matches.

- **A — Incorrect:** Dense embeddings do not guarantee lossless reconstruction.
- **B — Incorrect:** Retrieval still requires a ranking or similarity method.
- **C — Incorrect:** Similarity does not require identical vectors.
- **D — Correct:** Embedding models can encode meaning beyond exact word matches.

### 6. D
RAG supplies updated external evidence at inference time.

- **A — Incorrect:** This does not automatically supply current policies.
- **B — Incorrect:** Vocabulary size does not provide current facts.
- **C — Incorrect:** Sampling diversity does not update knowledge.
- **D — Correct:** RAG supplies updated external evidence at inference time.

### 7. C
Multiple demonstrations are supplied in context.

- **A — Incorrect:** No parameters are updated.
- **B — Incorrect:** Zero-shot uses no demonstrations.
- **C — Correct:** Multiple demonstrations are supplied in context.
- **D — Incorrect:** No student is trained from a teacher.

### 8. C
Low-rank trainable updates adapt a largely frozen base model.

- **A — Incorrect:** It updates the base model parameters.
- **B — Incorrect:** It prepares retrieval content rather than adapting weights.
- **C — Correct:** Low-rank trainable updates adapt a largely frozen base model.
- **D — Incorrect:** It selects output tokens without training.

### 9. A
This reduces dependence on a single split.

- **A — Correct:** This reduces dependence on a single split.
- **B — Incorrect:** Each fold requires fitting a model.
- **C — Incorrect:** No validation method guarantees this.
- **D — Incorrect:** A held-out test set should remain separate.

### 10. C
It predicts subsequent tokens using preceding context.

- **A — Incorrect:** It clusters vectors.
- **B — Incorrect:** It reduces dimensions.
- **C — Correct:** It predicts subsequent tokens using preceding context.
- **D — Incorrect:** It is commonly used for representation tasks.

### 11. D
An embedding lookup maps discrete IDs to continuous representations.

- **A — Incorrect:** They are numerical vectors.
- **B — Incorrect:** They have different representations.
- **C — Incorrect:** IDs are discrete identifiers.
- **D — Correct:** An embedding lookup maps discrete IDs to continuous representations.

### 12. D
Clear constraints and demonstrations guide formatting.

- **A — Incorrect:** This is unrelated to inference-time prompting.
- **B — Incorrect:** This removes needed constraints.
- **C — Incorrect:** This may increase variability.
- **D — Correct:** Clear constraints and demonstrations guide formatting.

### 13. C
Lower temperature sharpens the distribution.

- **A — Incorrect:** Sampling cannot guarantee truth.
- **B — Incorrect:** Context capacity is separate.
- **C — Correct:** Lower temperature sharpens the distribution.
- **D — Incorrect:** Decoding does not train weights.

### 14. D
It penalizes low probability assigned to the observed next token.

- **A — Incorrect:** BLEU is primarily an evaluation metric.
- **B — Incorrect:** Token IDs are categorical, not ordinal targets.
- **C — Incorrect:** This evaluates clustering.
- **D — Correct:** It penalizes low probability assigned to the observed next token.

### 15. A
Retrieval succeeded; the generator may not use its evidence.

- **A — Correct:** Retrieval succeeded; the generator may not use its evidence.
- **B — Incorrect:** The correct passage was already found.
- **C — Incorrect:** It does not directly address evidence use.
- **D — Incorrect:** This would remove retrieval capability.

### 16. B
It provides GPU-based DataFrame operations.

- **A — Incorrect:** It provides NLP processing tools.
- **B — Correct:** It provides GPU-based DataFrame operations.
- **C — Incorrect:** It optimizes neural network inference.
- **D — Incorrect:** It focuses on machine learning algorithms.

### 17. D
It provides GPU-accelerated ML algorithms.

- **A — Incorrect:** It primarily manipulates tabular data.
- **B — Incorrect:** It serves models.
- **C — Incorrect:** It represents models for exchange.
- **D — Correct:** It provides GPU-accelerated ML algorithms.

### 18. B, D
Compatible engines can import the standardized graph. ONNX supports interoperability across tools.

- **A — Incorrect:** Runtime and hardware affect speed.
- **B — Correct:** Compatible engines can import the standardized graph.
- **C — Incorrect:** Conversion needs validation.
- **D — Correct:** ONNX supports interoperability across tools.
- **E — Incorrect:** ONNX is a model format.

### 19. A
Combining compatible operations can reduce intermediate overhead.

- **A — Correct:** Combining compatible operations can reduce intermediate overhead.
- **B — Incorrect:** This is data collection.
- **C — Incorrect:** This is not engine optimization.
- **D — Incorrect:** Dropout is normally disabled for inference.

### 20. C
It provides model serving across supported backends.

- **A — Incorrect:** It provides numerical array operations.
- **B — Incorrect:** It processes DataFrames.
- **C — Correct:** It provides model serving across supported backends.
- **D — Incorrect:** It implements ML algorithms.

### 21. C
It packages supported models for inference deployment.

- **A — Incorrect:** It is an NLP library.
- **B — Incorrect:** It provides graph analytics.
- **C — Correct:** It packages supported models for inference deployment.
- **D — Incorrect:** It processes tabular data.

### 22. D
It provides these NLP pipeline components.

- **A — Incorrect:** It visualizes data.
- **B — Incorrect:** It handles tabular data.
- **C — Incorrect:** It handles numerical arrays.
- **D — Correct:** It provides these NLP pipeline components.

### 23. C
Vector search locates relevant records and metadata identifies sources.

- **A — Incorrect:** These are operational metrics.
- **B — Incorrect:** Vocabulary is not the document index.
- **C — Correct:** Vector search locates relevant records and metadata identifies sources.
- **D — Incorrect:** Loss does not identify document content.

### 24. B, C
Similarity requires compatible vector spaces. Migration should be checked on representative queries.

- **A — Incorrect:** Matching shape does not align semantic spaces.
- **B — Correct:** Similarity requires compatible vector spaces.
- **C — Correct:** Migration should be checked on representative queries.
- **D — Incorrect:** This does not fix vector compatibility.
- **E — Incorrect:** The spaces may be incompatible.

### 25. C
These capture responsiveness and reliability.

- **A — Incorrect:** It does not measure live service behavior.
- **B — Incorrect:** It does not measure user experience.
- **C — Correct:** These capture responsiveness and reliability.
- **D — Incorrect:** It does not measure response performance.

### 26. D
Batching can improve utilization but has latency and memory costs.

- **A — Incorrect:** Larger batches can require more memory.
- **B — Incorrect:** Batch size does not guarantee accuracy changes.
- **C — Incorrect:** The metrics can trade off.
- **D — Correct:** Batching can improve utilization but has latency and memory costs.

### 27. C
This handles transient failures while limiting retry storms.

- **A — Incorrect:** This can overload the service.
- **B — Incorrect:** Transport failures do not imply training defects.
- **C — Correct:** This handles transient failures while limiting retry storms.
- **D — Incorrect:** This hides failures.

### 28. C
It is reserved for final evaluation.

- **A — Incorrect:** It is not independent of development.
- **B — Incorrect:** It fits model parameters.
- **C — Correct:** It is reserved for final evaluation.
- **D — Incorrect:** It guides model selection.

### 29. A
The model fits training data increasingly well but generalizes worse.

- **A — Correct:** The model fits training data increasingly well but generalizes worse.
- **B — Incorrect:** Training loss is improving while validation worsens.
- **C — Incorrect:** The pattern alone does not establish leakage.
- **D — Incorrect:** Rising validation loss suggests otherwise.

### 30. B
Early stopping limits training beyond the best validation point.

- **A — Incorrect:** This compromises independent evaluation.
- **B — Correct:** Early stopping limits training beyond the best validation point.
- **C — Incorrect:** This removes evidence of generalization.
- **D — Incorrect:** This may favor overfitting.

### 31. A
Each epoch has 12 batches; three epochs yield 36 updates.

- **A — Correct:** Each epoch has 12 batches; three epochs yield 36 updates.
- **B — Incorrect:** This counts processed examples, not updates.
- **C — Incorrect:** This multiplies batch size by epochs.
- **D — Incorrect:** This is the count for one epoch.

### 32. A
An ablation isolates the component contribution.

- **A — Correct:** An ablation isolates the component contribution.
- **B — Incorrect:** This confounds effects.
- **C — Incorrect:** Different inputs reduce comparability.
- **D — Incorrect:** That does not isolate retrieval reranking.

### 33. C
Concurrent random assignment supports fair comparison.

- **A — Incorrect:** Temporal changes can bias comparison.
- **B — Incorrect:** Selective sampling biases results.
- **C — Correct:** Concurrent random assignment supports fair comparison.
- **D — Incorrect:** User differences confound results.

### 34. C
Perplexity summarizes predictive uncertainty on the evaluated text.

- **A — Incorrect:** Likelihood is not factual verification.
- **B — Incorrect:** Perplexity is not a hardware metric.
- **C — Correct:** Perplexity summarizes predictive uncertainty on the evaluated text.
- **D — Incorrect:** Decoding diversity depends on other settings.

### 35. D
Reviewers can judge semantic fidelity and coherence.

- **A — Incorrect:** This measures hardware state.
- **B — Incorrect:** This is a training configuration.
- **C — Incorrect:** Length does not establish quality.
- **D — Correct:** Reviewers can judge semantic fidelity and coherence.

### 36. B
Preferences can train reward models or support preference optimization.

- **A — Incorrect:** Preferences are not hardware instructions.
- **B — Correct:** Preferences can train reward models or support preference optimization.
- **C — Incorrect:** Tokenization remains necessary.
- **D — Incorrect:** Human preferences may themselves contain bias.

### 37. A, D
Quantization can change both quality and resource use. Controlled comparisons isolate precision effects.

- **A — Correct:** Quantization can change both quality and resource use.
- **B — Incorrect:** Size alone does not establish quality or speed.
- **C — Incorrect:** Reduced precision may affect outputs.
- **D — Correct:** Controlled comparisons isolate precision effects.
- **E — Incorrect:** Different workloads confound comparison.

### 38. B
These preserve key experimental conditions.

- **A — Incorrect:** This loses experimental context.
- **B — Correct:** These preserve key experimental conditions.
- **C — Incorrect:** This removes comparison artifacts.
- **D — Incorrect:** This prevents tracing variability.

### 39. D
The middle ranked value is robust to extreme observations.

- **A — Incorrect:** Extreme values can shift it substantially.
- **B — Incorrect:** It is directly determined by the largest value.
- **C — Incorrect:** It depends on extremes.
- **D — Correct:** The middle ranked value is robust to extreme observations.

### 40. C
It displays paired numerical observations.

- **A — Incorrect:** It hides individual relationships.
- **B — Incorrect:** It displays parts of a whole.
- **C — Correct:** It displays paired numerical observations.
- **D — Incorrect:** It does not show paired latency values.

### 41. C
Preprocessing statistics must be learned without validation leakage.

- **A — Incorrect:** This leaks target information.
- **B — Incorrect:** This uses validation information.
- **C — Correct:** Preprocessing statistics must be learned without validation leakage.
- **D — Incorrect:** This compromises test independence.

### 42. D
Recall for prohibited messages is zero.

- **A — Incorrect:** Imbalance alone does not imply corruption.
- **B — Incorrect:** Accuracy is a classification metric.
- **C — Incorrect:** It misses all positive examples.
- **D — Correct:** Recall for prohibited messages is zero.

### 43. B
The percentile describes the upper boundary for most observations.

- **A — Incorrect:** A percentile is not a sample count.
- **B — Correct:** The percentile describes the upper boundary for most observations.
- **C — Incorrect:** Mean and percentile are different.
- **D — Incorrect:** The percentile does not describe each request.

### 44. B
Precision is true positives divided by predicted positives.

- **A — Incorrect:** This is the false discovery proportion here.
- **B — Correct:** Precision is true positives divided by predicted positives.
- **C — Incorrect:** Precision is computable from the provided counts.
- **D — Incorrect:** This reverses the ratio.

### 45. B
Correlation describes association, not proof of causation.

- **A — Incorrect:** Generalization requires additional evidence.
- **B — Correct:** Correlation describes association, not proof of causation.
- **C — Incorrect:** Direction of causality is not established.
- **D — Incorrect:** Other factors may contribute.

### 46. D
This reduces privacy risk and respects data-use requirements.

- **A — Incorrect:** This may expose personal information.
- **B — Incorrect:** Unnecessary retention increases exposure.
- **C — Incorrect:** Accessibility does not establish permission.
- **D — Correct:** This reduces privacy risk and respects data-use requirements.

### 47. C
Disaggregated evaluation helps identify sources of disparity.

- **A — Incorrect:** Aggregate metrics can hide disparities.
- **B — Incorrect:** This does not directly diagnose bias.
- **C — Correct:** Disaggregated evaluation helps identify sources of disparity.
- **D — Incorrect:** This may worsen representation and fairness.

### 48. C, D
Trust boundaries and authorization limit the impact of injected instructions. Least privilege limits the impact of malicious instructions.

- **A — Incorrect:** Retrieved content is not trusted authority.
- **B — Incorrect:** Relevance does not imply safety.
- **C — Correct:** Trust boundaries and authorization limit the impact of injected instructions.
- **D — Correct:** Least privilege limits the impact of malicious instructions.
- **E — Incorrect:** More context does not enforce authorization.

### 49. B
Traceable evidence and uncertainty improve verifiability.

- **A — Incorrect:** Randomness does not ensure factual accuracy.
- **B — Correct:** Traceable evidence and uncertainty improve verifiability.
- **C — Incorrect:** Confidence does not establish correctness.
- **D — Incorrect:** This reduces transparency.

### 50. D
Guardrails reduce risk but need evaluation and complementary controls.

- **A — Incorrect:** Monitoring remains necessary.
- **B — Incorrect:** No guardrail guarantees perfect safety.
- **C — Incorrect:** Authorization must be enforced independently.
- **D — Correct:** Guardrails reduce risk but need evaluation and complementary controls.
