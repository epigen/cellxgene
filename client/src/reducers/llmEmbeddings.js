/*
  Reducers for LLMEmbedding.
  NOTE: could be refactored, by merging most actions
*/
const LLMEmbedding = (
  state = {
    messages: "Welcome to CellWhisperer! To get started, please try some of the following options:\n\n1. Highlight your cells of interest, e.g. by typing \"/search NK cells\" or \"/search inflammation\" in the text box below\n\n2. Select a group of cells, either by drawing a line around them, or via the search term histogram on the bottom left. Then, press the “Describe the selected pseudocell” button.\n\n3. Enter questions about your selected cells into the chat box, for example \"What distinguishes these cells from macrophages?\"\n\n4. You can also ask general questions, for example \"What is the role of IL-2R in natural killer cells?\"\n\n5. If a comment does not make sense to you, you can press the 👎 icon, and CellWhisperer will generate a new response.\n\n6. You can help us improve CellWhisperer by pressing the 👍 icon for answers that appear to be correct and useful.\n\n \n\nPlease keep in mind that CellWhisperer is an AI system and may produce incorrect or misleading results. CellWhisperer is best used as a tool for data exploration and hypothesis generation.",
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
