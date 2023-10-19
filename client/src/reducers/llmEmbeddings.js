/*
  Reducers for LLMEmbedding.
*/
const LLMEmbedding = (
  state = {
    outputText: "",
    inProgress: false,
  },
  action
) => {
  switch (action.type) {
    case "send request text to embedding model": {
      return {
        ...state,
        inProgress: true,
      };
    }
    case "embedding model annotation response from text": {
      return {
        ...state,
        outputText: action.data, // TODO should be cells and add a continuousnnotation
        inProgress: false,
      };
    }

    case "send request cells to embedding model": {
      // TODO not necessary probably
      return {
        ...state,
        inProgress: true,
      };
    }
    case "embedding model text response from cells": {
      return {
        ...state,
        outputText: action.data,
        inProgress: false,
      };
    }

    default:
      return state;
  }
};

export default LLMEmbedding;
