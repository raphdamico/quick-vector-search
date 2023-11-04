// Import the necessary functions from transformers library
import { env, pipeline, cos_sim } from "https://cdn.jsdelivr.net/npm/@xenova/transformers";
env.allowLocalModels = true;
env.allowRemoteModels = true;
env.useBrowserCache = true;

async function loadModel() {
    // Ensure that 'sentence-transformers/all-MiniLM-L6-v2' is a valid model path for the transformers library you are using
    const model = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    return model;
}

export async function compareSentences(search_sentence, sentences_to_compare) {

  // Load model
  const t_start = new Date().getTime();
  console.log(`Start`)
  const sentences = [search_sentence, ...sentences_to_compare];
  const model = await loadModel();
  
  // Calculate vectors
  const t_modelLoaded = new Date().getTime();
  console.log(`Model load time ${((t_modelLoaded - t_start) / 1000 ).toFixed(2)}s`)
  let output = await model(sentences, { pooling: 'mean', normalize: true });  
  output = output.tolist();
  
  // Calculate cosine similarity 
  const t_calculatedVectors = new Date().getTime();
  console.log(`Processing vectors time ${((t_calculatedVectors - t_modelLoaded)/1000).toFixed(2)}s`)
  let similarities = []
  for (let i=0; i<sentences_to_compare.length; i++) {
    similarities[i] = {
      sentence: sentences_to_compare[i],
      score: cos_sim(output[0].flat(), output[i+1].flat())
    }
  }
  
  // Return output
  const t_calculatedCosSim = new Date().getTime();
  console.log(`Cosine similarity ${((t_calculatedCosSim - t_calculatedVectors)/1000).toFixed(2)}s`)  
  console.log(`Total time ${((t_calculatedCosSim - t_start)/1000).toFixed(2)}s`)  
  return similarities;
}
