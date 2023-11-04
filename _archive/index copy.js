// Import the necessary functions from transformers library
import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers";

env.allowLocalModels = false;

async function loadModel() {
    // Ensure that 'sentence-transformers/all-MiniLM-L6-v2' is a valid model path for the transformers library you are using
    const model = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    return model;
  }
  
  
// Function to calculate cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
  let dotproduct = 0;
  let mA = 0;
  let mB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotproduct += (vecA[i] * vecB[i]);
    mA += (vecA[i] * vecA[i]);
    mB += (vecB[i] * vecB[i]);
  }
  mA = Math.sqrt(mA);
  mB = Math.sqrt(mB);
  const similarity = (dotproduct) / ((mA) * (mB));
  return similarity;
}

// Function to encode sentences and calculate similarity
async function compareSentences(sentence1, sentence2) {
    const model = await loadModel();
    
    // Obtain the embeddings from the model
    const embeddings1 = await model(sentence1);
    const embeddings2 = await model(sentence2);
  
    console.log('Embeddings1:', embeddings1);
    console.log('Embeddings2:', embeddings2);
  
    // Check if embeddings are non-empty arrays
    if (!Array.isArray(embeddings1) || !embeddings1.length || !Array.isArray(embeddings2) || !embeddings2.length) {
      console.error('One of the embeddings is not a non-empty array');
      return;
    }
  
    // Flatten the embeddings in case they are arrays of arrays
    const flatEmbedding1 = embeddings1.flat();
    const flatEmbedding2 = embeddings2.flat();
  
    // Check if the flattened embeddings are valid
    if (!flatEmbedding1.length || !flatEmbedding2.length) {
      console.error('One of the flattened embeddings is empty');
      return;
    }
  
    // Calculate the average embeddings if necessary
    const averageEmbedding1 = flatEmbedding1[0].length ? flatEmbedding1.reduce((acc, val) => acc.map((num, idx) => num + val[idx]), new Array(flatEmbedding1[0].length).fill(0)).map(el => el / flatEmbedding1.length) : flatEmbedding1;
    const averageEmbedding2 = flatEmbedding2[0].length ? flatEmbedding2.reduce((acc, val) => acc.map((num, idx) => num + val[idx]), new Array(flatEmbedding2[0].length).fill(0)).map(el => el / flatEmbedding2.length) : flatEmbedding2;
  
    console.log('Average Embedding1:', averageEmbedding1);
    console.log('Average Embedding2:', averageEmbedding2);
  
    // Calculate the cosine similarity between the average embeddings
    const similarity = cosineSimilarity(averageEmbedding1, averageEmbedding2);
    if (isNaN(similarity)) {
      console.error('The cosine similarity calculation resulted in NaN');
      return;
    }
  
    console.log(`The similarity score between the sentences is: ${similarity}`);
    return similarity;
  }
  
  

console.log("Run it")
compareSentences('This is a sentence.', 'This is a similar sentence.')
    .then(similarity => console.log(`Similarity: ${similarity}`))
    .catch(error => console.error(error));
console.log("It ran!")

export { compareSentences };
