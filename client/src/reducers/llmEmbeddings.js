/*
  Reducers for LLMEmbedding.
*/
const LLMEmbedding = (
  state = {
    outputText: "",
    loading: false,
  },
  action
) => {
  switch (action.type) {
    case "request to embedding model started": {
      return {
        ...state,
        loading: true,
      };
    }
    case "embedding model annotation response from text": {
      return {
        ...state,
        loading: false,
      };
    }

    case "embedding model text response from cells": {
      return {
        ...state,
        outputText: action.data,
        loading: false,
      };
    }
    case "request llm embeddings error": {
      return {
        ...state,
        outputText: `ERROR: ${action.error}`,
        loading: false,
      };
    }

    default:
      return state;
  }
};

export default LLMEmbedding;
