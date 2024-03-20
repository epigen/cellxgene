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

    case "chat request start": {
      return {
        ...state,
        loading: true,
        // error: null,
      };
    }
    case "chat request success": {
      return {
        ...state,
        outputText: action.payload.text, // Assuming the payload contains a 'text' field with the response
        loading: false,
        // error: null,
      };
    }
    case "chat request failure": {
      return {
        ...state,
        outputText: action.payload, 
        loading: false,
        // error: action.payload, // Error message
      };
    }

  default:
    return state;
  }
};

export default LLMEmbedding;
