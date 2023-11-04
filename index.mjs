// Import the necessary functions from transformers library
import { env, pipeline, cos_sim } from "https://cdn.jsdelivr.net/npm/@xenova/transformers";
env.allowLocalModels = false;

async function loadModel() {
    // Ensure that 'sentence-transformers/all-MiniLM-L6-v2' is a valid model path for the transformers library you are using
    const model = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    return model;
}

export async function compareSentences(search_sentence, sentences_to_compare) {

    const sentences = [search_sentence, ...sentences_to_compare];
    const model = await loadModel();
    let output = await model(sentences, { pooling: 'mean', normalize: true });
    output = output.tolist();
    let similarities = []
    for (let i=0; i<sentences_to_compare.length; i++) {
      similarities[i] = {
        sentence: sentences_to_compare[i],
        score: cos_sim(output[0].flat(), output[i+1].flat())
      }
    }
    return similarities;
}
