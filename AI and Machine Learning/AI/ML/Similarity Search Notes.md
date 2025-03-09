## What is similarity search?

Imagine you have an image, and you want to find other images that might be similar to the one you have. This is essentially what similarity search algorithms do, but we aren't just limited to images. We can apply similarity search to a range of modalities such as text, audio, videos, or anything else that can be represented as data.

## Use Cases

- **Product recommendation systems** - It can be used to recommend products similar to previously bought items by finding similar factors such as descriptions, brands, and price. Very useful for e-commerce setups.
- **Search engines** - Retrieving documents similar to a query (this is one of the ways certain RAG systems work).
- **Anomaly detection** - Find data points that are dissimilar from the rest.
- **Image and video retrieval** - Find images and videos that are similar to an input query (newer multimodal models allow for image to video retrievals and vice versa).

## Data Preparation

- **Data gathering** - Collect the initial data that you want to be able to search through. File format used is very important. 
	- For unstructured data, plain text (.txt) and markdown (.md) are your go-to types. 
	- JSON and HTML files are also good for data which is a little more structured. 
	- A lot of models also support PDF and Word documents (.docx), but these will need some pre-processing to make them more effective. Libraries such as PyPDF2 or python-docx are useful for this although you still might not get very effective results compared to plain text and markdown. 
	- Images, video, and audio data require more specialised processing to extract meaningful features. Audio files can be transcribed to text using speech-to-text APIs, and videos are usually processed by extracting keyframes and using image/video embedding models.
- **Pre-processing** - Clean and prepare your gathered data. It's worth noting that this isn't always required and may actually hurt similarity search accuracy. You should benchmark results with both processed and none processed inputs to find what works best for you.
	- For text, removing punctuation, converting text to lowercase, and stemming are usually the first things to look into. But be aware that a lot of text embedding models are trained on data which does not have these optimisations so may perform better on raw un-processed inputs. If using Python, the NLTK package is a personal recommendation to do the heavy lifting for you.
	- For images, resizing and normalising the pixel values is useful.
	- For audio, always convert into a standardised format such as WAV, FLAC, or mp3.
		- WAV is an uncompressed, lossless format which means it retains all the original audio data at the expense of file size. The downside is that the file sizes are usually larger than alternatives which can lead to storage issues, as well as processing potentially taking longer and becoming more expensive.
		- FLAC isn't as common but provides a nice middle ground between WAV and mp3. It's a lossless format similar to WAV, but does use compression to make the file size a little bit smaller while maintaining audio quality. This format probably isn't as widely supported as alternatives though.
		- mp3 is probably the most common hence it being widely supported by a lot of audio capable models. The biggest downside is that it uses lossy compression, so you'll lose data, leading to worsened similarity search performance.

## Vector Embeddings

### What is a vector and why is it useful?

In simple terms, a vector is just a list of numbers. They're great for storing information in a structured, numerical format. Generally in computational setups, they're represented as arrays of floats (decimal numbers). Common vector embedding models will output numerical values that have been normalised (made to be a number between -1 and 1). Vectors will come with varying levels of "dimensionality". You can think of higher dimensionality as allowing for more capacity for capturing "meaning". Although it's worth noting that adding more dimensions to a vector will not always mean more "meaning" will be captured, it just allows more capacity to do so. Dimensionality of a vector is equal to the length of an array, for example, 

````javascript
 [0.23, 0.35, 0.21] // 3 dimensional vector
 [0.98, 0.64, 0.22, 0.34] // 4 dimensional vector
````

Popular text embedding models such as Gemini's `text-embedding-004` will output vector embeddings with dimensions as high as 768 (meaning an array with 768 elements). This is what allows the model to be able to capture meaning of language in so much depth.

Using vectors for similarity search is incredibly useful, since it allows us to apply maths to the meaning of words to find words and sentences that are similar to others by finding the distances between them in a vector space, since their vectors will be very close together if they mean similar things.

In more complex data such as images, and videos, we can apply similar logic to features such as pixel colour values, textures and shapes, as well as any text transcribed by an audio-to-text or image-to-text model.

For anyone who has knowledge on more traditional physics/geometry, classical vectors usually consist of two numbers, a direction, and a magnitude (also known as length). For vectors of small dimensionality, such as 2D or 3D, it's fairly simple to use these vectors to draw arrows, or plot points on a graph (also known as a vector space).  We can then use those to calculate "similarity" by either calculating the distance between points, or the angle between them. For higher dimensionality vectors (> 3) it becomes harder to visualise them, but we can still apply the same techniques in a (much more complex) vector space.

### Embedding Models

Embedding models work by learning to capture the semantic meaning of your data. Similar items will be given vectors that are close to each other in vector space.

Generally, the model that will be the best for you will depend on your use case, and the input data you have.

For text you have two main options:
#### Word embeddings

- Word embedding models will focus on generating vectors for individual words based on their meanings within a corpus of text.
- Words that appear in similar contexts across the corpus of text will be given vectors close together in a vector space.
- They're generally trained to predict what the surrounding words will be.
- Popular options include: Word2Vec, GloVe, and FastText.
##### Use cases:
- Word similarity: Useful for finding words that are semantically similar, for example "king", and "queen".
- Text classification: Great for classifying and finding named entities such as people, organisations, or locations.
- Machine translation: Great for aligning words between multiple languages.
##### Limitations:
- Word embeddings represent individual words, not entire sentences or documents.
- They won't capture the overall meaning of a sentence, especially when they contain complex syntax and semantics.
- They won't take into account the order of words within a sentence.

#### Sentence embeddings

- Sentence embeddings represent entire sentences or paragraphs as vector (instead of individual words).
- They can capture complete overall meaning and context of sentences.
- Great for capturing nuances of languages such as sarcasm or irony. Word embeddings struggle to do this effectively.
- Popular options include: [Sentence-BERT models](https://www.sbert.net/), and models from large companies such as Google's `text-embedding-005` and OpenAI's `text-embedding-3-small` and `text-embedding-3-large`.
##### Use cases:
- Semantic search: Great for finding sentences which are similar to others. This allows for more comprehensive vectors for tasks like QA matching, which we've used to optimise both cost and latency of RAG pipelines. 
- Sentiment analysis: Very useful for measuring the emotional meaning of sentences and assigning them scores.
##### Limitations:
- Usually more computationally expensive than word embedding models, leading to higher cost and latency.
- The higher cost and latency will scale with the amount of data you're trying to embed.

Personally, if you're unsure of which to use, start with sentence embedding models and test if they work for you.

For image embeddings you can use models such as ResNet, EfficientNet, or OpenAI's CLIP. It's worth noting that CLIP is also capable of text embeddings.

### Embedding Models vs LLMs: Understanding the Difference

While both embedding models and LLMs are related to NLP, and use neural networks, their core functionality differ significantly. 

**Purpose** - Embedding models focus on converting data such as text and images into what we call dense vector representations that capture the semantic meaning of words, sentences, or entire documents. LLMs are used to generate human-like text, answer questions, and generally focus on generating contextually relevant text.

**Key functionality** 
- Embedding models: similarity search and clustering.
- LLMs: text generation, question answering, and language translation.

**Training** - This is where the confusion usually lies.
- Embedding models: trained to learn semantic representations that might group items together in a vector space.
- LLMs: Trained on datasets of text to learn patterns and relationships in language.

A good thing to know is that vector embeddings are used in the LLM text generation process. More about this will be included in the further reading section at the end.

## Measuring Similarity

Now that we have data that's represented as vector embeddings, we need a way to actually measure the similarity between them. 
### Common metrics

#### Euclidean Distance

A straight line drawn between two points in a vector space. The lower the distance, the more similar in meaning the embeddings are. Only suitable for embeddings of low dimensionality. Good for similarity searches where magnitude (length) is important.

```javascript
 const euclideanDistance = (vector1, vector2) => { 
	 if (vector1.length !==  vector2.length) { 
	   throw new Error("Vectors must have the same length"); 
	 } 
	 
	 let sumOfSquares = 0; 
	 for (let i = 0; i < vector1.length; i++) { 
	   sumOfSquares += Math.pow(vector1[i] - vector2[i], 2);
	 } 
	 
	 return Math.sqrt(sumOfSquares); 
 }
```

#### Cosine Similarity (Best when working with vector embeddings)

A measure of the cosine of an angle between two vectors. Generally ranges between -1 and 1 where -1 is complete opposite in meaning and 1 is perfect similarity (the exact same words in both syntax and word order). In many practical applications. it's rare you'll find any negative values, since most vectors won't have negative components. In these situations 0 can be considered to be completely dissimilar. Cosine sim is great for situations where magnitude isn't important and is my personal recommendation when working with vector embeddings, since most transformer based models will normalise the vectors magnitudes to 1 for you by default. The cosine similarity calculation might look like this in code:

```javascript
  // Calculate cosine similarity between user query and mock queries
  const similarities = matrices.map(
    (matrix) =>
      userQueryMatrix.dot(matrix.transpose()) /
      (userQueryMatrix.norm() * matrix.norm()),
  );
```

This code loops through an array of vector embeddings that have been converted to matrices and performs a cosine similarity calculation between each embedding and the embedded version of a user input. The similarity with the highest value will be the matrix with the smallest angle between them.

You can think of the angle to value looking something like this:

**0 degrees** - vectors are perfectly aligned, the cosine similarity is 1 and is a perfect string match.
**90 degrees** - vectors are at a right angle to each other (also known as orthogonal), and the cosine similarity will be 0.
**180 degrees** - vectors are complete opposites to each other, and the cosine similarity will be -1.

Most similarities will fall somewhere between 0 and 1 (between 0-90 degrees).

One of the biggest downsides to cosine similarity is that it's heavily dependant on the quality of vector embeddings. This is why I benchmarked a range of different embedding models to see what worked best for our use-case. More on this can be found in the benchmarks section.

#### Dot Product 

The dot product is similar to cosine similarity in the sense that it measures the angle between two vectors. A good way to think about it is that it checks whether two vectors are pointing in the same direction. The dot product equation looks like this:

```
A ⋅ B = |A| * |B| * cos(θ)

Where |A| and |B| are the magnitudes (lengths) of vectors A and B and θ is the angle between the two vectors.
```

However, since dot product is dependent on the vectors magnitudes (lengths), if they're normalised to 1, which is what most transformer based models do by default (such as Gemini and S-BERT), the dot product becomes equal to the cosine of the angle between them. This means that dot product becomes equivalent to cosine similarity.

## How does this all relate to the Docusaurus sentence similarity research I've been working on?

Back when we were working on the Docusaurus LLM to answer questions on our developer documentation, I realised that we could optimise our RAG workflow by caching responses to questions that are commonly asked. This would mean that we could skip the LLM entirely, reducing both cost and latency. For this, I created a very small set of mock QA values. This was structured as an array of objects with `cachedQuery` and `llmResponse` fields. 

```typescript
type TMockData = {
  cachedQuery: string;
  llmResponse: string;
};
 
/*
After updating these values, run the following command to update the generated embeddings:
`npm run generate-embeddings`
*/
export const mockData: TMockData[] = [
  {
    cachedQuery: "How do I handle unsupported locale pages in my application?",
    llmResponse: "To handle unsupported locale pages, you can implement type narrowing in TypeScript to check if the page is of type PageDynamic or PageDefault and if it has the isPageForUnsupportedLocale field set to true. If either condition is met, you should render a 404 not found page. Here’s an example: ",
  },
  {
    cachedQuery: "Can you explain how to filter unsupported articles in my application?",
    llmResponse: "To filter out articles that are marked as unsupported locales, you can use the following code snippet:",
  },
  {
    cachedQuery: "How do I set up the codebase locally?",
    llmResponse: "To set up the codebase locally, follow these steps:",
  },
  {
    cachedQuery: "How does the do not track option work with our videos?",
    llmResponse: "The 'Do Not Track' (DNT) option for videos works by checking user consent before enabling tracking features.",
  },
  {
    cachedQuery: "What is our preferred payment option?",
    llmResponse: "Stripe, but Ayden is also a strong contender.",
  },
];
```

In theory, we could take a very simple approach and do some exact string matching. So if the user input string is strictly equal to one of the strings we have in our mock data, we can just send the cached LLM response back. This is great, and would work, but has some significant limitations in terms of how often a user would actually input an exact string. To combat that we could create a larger database of variations of the same sentences, but there are better ways to deal with similarity searching in QA tasks.

We could theoretically make the string matching system more comprehensive by using some kind of lexical similarity algorithm. Popular options for this include:
- **Levenshtein Distance** -  This measures the minimum number of single-character edits (additions, deletions, substitutions) it would take to change one word, or sentence to another. A lower Levenshtein distance indicates higher similarity. For example, "kitten" and "sitting" have a distance of 3, since you'd need to change 3 characters to go from one to another.
- **Jaccard Similarity** -  This measures the similarity between two sets. In the context of strings, these sets are often sets of characters or words. The similarity is calculated as the size of the intersection of the sets divided by the size of their union. For example, if you're comparing two strings, you might break them down into sets of characters or n-grams (sequences of n characters). The Jaccard similarity would then tell you how much overlap there is between those sets. 
- **Jaro Similarity** - This is a measure of similarity between two strings, particularly useful for short strings like names. It considers the number of matching characters and transpositions (swapped characters). The Jaro-Winkler distance is a variation that gives more weight to matching prefixes, making it even more suitable for short strings.

These are great for tasks such as finding structural similarities and fuzzy matching. We previously used lexical similarity for some of the assistants fuzzy matching functionality back on the eJ PoC. Lexical similarity starts to break down when we need to know the difference in **semantic meaning** instead of just basic character-level differences. For example, "How do I reset my branch in git?" and "How do I revert my branch in git?." have a very high lexical similarity, even though the semantic meaning is completely different have have different commands. That's where we can start to push string similarity a bit further and try to identify strings that are similar to each other in meaning.

This is where vector embeddings and cosine similarity come in. Currently I've implemented Google's `text-embedding-004` to get embeddings for all the cached queries and a user input that we get from the front-end.

```typescript
 const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

 export const model = genAI.getGenerativeModel({ model: "text-embedding-004"});
  
  // Get embeddings for all cached queries
 const embeddings = await Promise.all(
   mockData.map((data) => model.embedContent(data.cachedQuery)),
 );
```

This will be swapped out for an OpenAI embedding model in the future. 

Using the embeddings created, we can then perform cosine similarity between the user input embeddings and the embeddings for each cached query to find the similarities between them all. In our current setup, we have it set to check if any similarities are above 0.9, and if so, use the cached response for the query with the highest similarity. If there are no similarities above 0.9, then we can send the user message to the LLM to get a new response.

We're also storing the vector embeddings for the cached queries so that we don't create unnecessary requests to the `text-embedding-004` model on every new POST request.

```typescript
 const similarityScores = await calculateCosineSimilarity(
   input.message,
   fs.existsSync(storedVectorEmbeddingsFilePath)
     ? loadEmbeddings()
     : await generateEmbeddings(mockData),
 );
```

It's worth noting that the system still needs to create embeddings for the user input every time (unless we were to cache these in the future too), but this is much cheaper and faster than a call to an LLM.

Currently we're just storing the embeddings as matrices in a JSON file. This is a rudimentary solution, but it works for our use case. Vector stores should be used for systems with more comprehensive requirements.

The only reason we're converting them to matrices is for library specific optimisations and not for any other reason related to the calculation.

In the future we can improve the system by collecting actual common queries and answers and building up a larger database of them. The more cached queries we have, the more likelihood of us being able to bypass the LLM with a cached response. 

### Embedding model benchmarks for QA matching tasks

I created a benchmark to test how well certain algorithms work alongside cosine similarity to find out which models I deemed to perform best.

For example, in my opinion the strongest performer was Gemini's embeddings + cosine similarity:

```javascript
Embeddings + Cosine Similarity Results:
{
  userQuery: 'How is the codebase set up to handle pages with unsupported locales?',
  cachedQuery: 'How do I handle unsupported locale pages in my application?',
  similarity: 0.8661192826917277,
  response: 'To handle unsupported locale pages, you can implement type narrowing in TypeScript to check if the page is of type PageDynamic or PageDefault and if it has the isPageForUnsupportedLocale field set to true. If either condition is met, you should render a 404 not found page. Here’s an example: '
}
{
  userQuery: 'How is the codebase set up to handle pages with unsupported locales?',
  cachedQuery: 'How do I set up the codebase locally?',
  similarity: 0.5198762740434945,
  response: 'To set up the codebase locally, follow these steps:'
}
{
  userQuery: 'How is the codebase set up to handle pages with unsupported locales?',
  cachedQuery: 'Can you explain how to filter unsupported articles in my application?',
  similarity: 0.5159442085159088,
  response: 'To filter out articles that are marked as unsupported locales, you can use the following code snippet:'
}
{
  userQuery: 'How is the codebase set up to handle pages with unsupported locales?',
  cachedQuery: 'How does the do not track option work with our videos?',
  similarity: 0.3747243367654625,
  response: "The 'Do Not Track' (DNT) option for videos works by checking user consent before enabling tracking features."
}
{
  userQuery: 'How is the codebase set up to handle pages with unsupported locales?',
  cachedQuery: 'What is our preferred payment option?',
  similarity: 0.2823486516729333,
  response: 'Stripe, but Ayden is also a strong contender.'
}
```

As you can see, the cached query with the highest similarity is lexically very different but still semantically very similar. It doesn't quite meet our requirement of 0.9 but the logic is there, and that similarity acceptance rate could be lowered depending on the project. 

As an example as to how lexically dissimilar the two queries are, here is the same benchmark using Levenshtein distance + cosine similarity instead:

```javascript
 Levenshtein Similarity Results:
 {
  userQuery: 'How is the codebase set up to handle pages with unsupported locales?',
  cachedQuery: 'How do I handle unsupported locale pages in my application?',
  similarity: 0.36764705882352944
}
{
  userQuery: 'How is the codebase set up to handle pages with unsupported locales?',
  cachedQuery: 'Can you explain how to filter unsupported articles in my application?',
  similarity: 0.17391304347826086
}
{
  userQuery: 'How is the codebase set up to handle pages with unsupported locales?',
  cachedQuery: 'How do I set up the codebase locally?',
  similarity: 0.3970588235294118
}
{
  userQuery: 'How is the codebase set up to handle pages with unsupported locales?',
  cachedQuery: 'How does the do not track option work with our videos?',
  similarity: 0.36764705882352944
}
{
  userQuery: 'How is the codebase set up to handle pages with unsupported locales?',
  cachedQuery: 'What is our preferred payment option?',
  similarity: 0.25
}
```

I also benchmarked a few of the S-BERT (sentence transformer) models, this time in Python, but Gemini's `text-embedding-004` still came out on top:

`mockData`:

```Python
mockData: List[Dict[str, str]] = [
    {"cachedQuery": "How do I set up the codebase locally?",
     "llmResponse": "To create a new branch, use 'git branch <branch-name>' or 'git checkout -b <branch-name>'"},
    {"cachedQuery": "What is the difference between git merge and git rebase?",
     "llmResponse": "Git merge combines changes from different branches, while rebase moves commits to a new base commit"},
    {"cachedQuery": "How can I undo the last commit in Git?",
     "llmResponse": "Use 'git reset HEAD~1' to undo the last commit but keep changes, or 'git reset --hard HEAD~1' to remove changes completely"},
    {"cachedQuery": "What does git stash do?",
     "llmResponse": "Git stash temporarily saves changes that are not ready to be committed, allowing you to switch branches"},
    {"cachedQuery": "How do I resolve merge conflicts in Git?",
     "llmResponse": "Open the conflicting files, manually edit the conflicts, stage the files, and then complete the merge"}
]
```

User query:

```python
 user_query = "How do I stash my changes in git?"
```

Benchmark results:

```Python
 Embedding Model Performance Benchmark:
 
 all-MiniLM-L6-v2:
 Average Similarity: 0.4746
 Individual Similarities:
   Query 1: 0.2351
   Query 2: 0.3960
   Query 3: 0.5785
   Query 4: 0.6393
   Query 5: 0.5243

 all-mpnet-base-v2:
 Average Similarity: 0.4306
 Individual Similarities:
   Query 1: 0.3858
   Query 2: 0.3213
   Query 3: 0.4272
   Query 4: 0.6269
   Query 5: 0.3919

 paraphrase-MiniLM-L3-v2:
 Average Similarity: 0.4652
 Individual Similarities:
   Query 1: 0.1377
   Query 2: 0.4488
   Query 3: 0.5181
   Query 4: 0.7813
   Query 5: 0.4403

 Gemini Embedding:
 Average Similarity: 0.6232
 Individual Similarities:
   Query 1: 0.4840
   Query 2: 0.5591
   Query 3: 0.6396
   Query 4: 0.8589
   Query 5: 0.5745
```

Query 4 was the most similar from a human perspective in this scenario, and Gemini seemed to identify that best, with `paraphrase-MiniLM-L3-v2` not far behind.

I plan to test the OpenAI embedding models in the same way in the future.
## Further reading

### How embeddings are used in LLM text generation

Vector embeddings are used in the LLM text generation process. After the initial tokenization process (input text being broken up into smaller units), each token will be converted into a vector embedding using something called an embedding matrix. The embedding matrix is learned during the training of an LLM. LLMs which use transformer architectures will then pass those embeddings into the encoding layers which contain an attention mechanism. The attention mechanism allows the model to weigh the importance of different words in the input sequence when generating the output.

#### Embedding matrix vs Embedding models

- **Embedding matrix** - This is essentially a lookup table, which is represented as a large matrix where each row corresponds to a token in the LLM's vocabulary. Each row will contain a corresponding vector embedding for that given token. The number of rows is always equal to the amount of vocabulary, e.g. 10,000 words = 10,000 rows in the matrix. These embeddings are learned during an LLMs training process.
- **Embedding models** - The difference here is that embedding models learn semantic meaning for a different scope and purpose. Embedding models will generally be trained to learn these relationships independently of any given downstream task. This means that the embeddings can generally be reused across a range of NLP tasks.

During the training of an LLM, it learns to find semantic relationships in words for the purpose of generating text, whereas embedding models learn relationships for a much more general purpose.

#### Decoder stage

Eventually it will get to a decoder stage where the LLM will create a probability distribution using something called a softmax function. The purpose of the softmax function is to take the raw scores from the LLM and turn them into output probabilities. This is where the `temperature` value (we can change this with our OpenAI assistants) comes into effect since it controls the general `sharpness` of the probability distribution. It is also where another thing we can change called `top_p` comes into effect. `top_p` works by sorting the probability scores that the LLMs output layer creates in a descending order, and can limit the possible tokens to the ones with higher probability values (meaning they're more likely to be outputted by the LLM, leading to less nonsensical and incoherent text). Both `temperature` and `top_p` are used to limit the "randomness" of a model, but only one of these values should be changed at a time. By changing them both, you're essentially adding two filters to the probability distribution, leading to unpredictable outputs and a loss of control over the LLMs behaviour.

### Visualising vector embeddings

Visualising lower dimensional data such as 2D and 3D is fairly trivial, since we can just plot them as points in a standard graph system, and draw lines between if using something like Euclidean distance to find similarities. Things start to get more complicated when we get into higher dimensions, since we as humas don't really have any way to visualise anything higher than 3D, let alone dimensions as high as 768+.

Luckily, we do have ways to reduce dimensionality down to 2D or 3D for visualisation reasons. Whilst I still need to do research into how any of these work, common methods are:

- Principle Component Analysis (PCA)
- t-Distributed Stochastic Neighbour Embedding (t-SNE)
- UMAP (Uniform Manifold Approximation and Projection)

Visualisation can be very useful since it allows us to cluster the vectors using methods such as K means clustering to extract meaningful insights like finding holes in our documentation, where questions are asked but we don't have cached responses for them, evaluating embedding quality, and detecting outliers that we might need to add more cached query variations for.

It can also be great for error analysis, providing us ways to evaluate LLM performance we we haven't really had a way to do before.

Finally, the number 1 most valuable insight would be for us to be able to understand user intent when asking questions. If this is applied to a chatbot in an e-commerce setting, it could have some significant positive sales implications. You could even use it to analyse trends, increasing demands, or products which may be losing sales due to lack of quality descriptions etc. These insights can even be used to pass into product recommendation systems, further driving potentially sales.