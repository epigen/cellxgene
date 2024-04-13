/*
  Reducers for LLMEmbedding.
  NOTE: could be refactored, by merging most actions
*/
const LLMEmbedding = (
  state = {
    messages: "Welcome to CellWhisperer! This Mini-Guide will get you started:\n\n-  Type \"/search ...\" to search for cells\n-  Select your cells of interest and start a conversation\n-  CellWhisperer may produce misleading and incorrect results.\n    -  Use it as hypothesis generator rather than as ground truth\n\n\nHelp us improve it: 👎 will generate a new response. If the response improved, please indicate so 👍",
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
        messages: action.data,
        loading: false,
      };
    }

    case "embedding model gene contributions response": {
      return {
        ...state,
        messages: action.data,
        loading: false,
      };
    }
    case "request llm embeddings error": {
      return {
        ...state,
        messages: `ERROR: ${action.error}`,
        loading: false,
      };
    }

    case "chat reset": {
      return {
        ...state,
        // Add an empty message to the end of the list of messages
        messages:  [],
        // error: null,
      };
    }

    case "chat request start": {
      return {
        ...state,
        loading: true,
        // Add an empty message to the end of the list of messages
        messages:  action.newMessages.concat({ value: "", from: "gpt" }),
        // error: null,
      };
    }
    case "chat request success": {
      return {
        ...state,
        // Replace the last entry of messages
        messages: state.messages.slice(0, -1).concat({ value: action.payload, from: "gpt" }),
        loading: false,
        // error: null,
      };
    }
    case "chat request failure": {
      return {
        ...state,
        messages: state.messages.slice(0, -1).concat({ value: `ERROR: ${action.payload}`, from: "gpt" }),
        loading: false,
        // error: action.payload, // Error message
      };
    }

  default:
    return state;
  }
};

export default LLMEmbedding;
