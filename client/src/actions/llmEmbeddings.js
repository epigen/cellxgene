import { doJsonRequest } from "../util/actionHelpers";
import * as globals from "../globals";

function fetchJson(pathAndQuery) {
  return doJsonRequest(
    `${globals.API.prefix}${globals.API.version}${pathAndQuery}`
  );
}

/*
LLM embedding querying
*/
export function requestEmbeddingLLMWithTextAction(text) {
  return async function requestEmbeddingLLMWithTextThunk(dispatch) {
    // , getState
    /*
      Send a request to the LLM embedding model with text
    */
    dispatch({
      type: "send request text to embedding model",
    });
    return fetchJson(`llmWithText?text=${text}`).then((response) =>
      dispatch({
        type: "embedding model annotation response from text",
        data: response,
      })
    );
  };
}

export function requestEmbeddingLLMWithCellsAction() {
  // TODO fetch cells
  const cells = "TODO";
  return async function requestEmbeddingLLMWithCellsThunk(dispatch) {
    // , getState
    /*
      Send a request to the LLM embedding model with cells
    */
    dispatch({
      type: "send request cells to embedding model",
    });
    return fetchJson(`llmWithCells?cells=${cells}`).then((response) =>
      dispatch({
        type: "embedding model text response from cells",
        data: response,
      })
    );
  };
}
