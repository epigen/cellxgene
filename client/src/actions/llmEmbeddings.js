import * as globals from "../globals";
import { annotationCreateContinuousAction } from "./annotation";
import { matrixFBSToDataframe } from "../util/stateManager/matrix";

/*
  LLM embedding querying
  NOTE: could be refactored by merging most dispatchers
*/
export const requestEmbeddingLLMWithCells =
  /*
    Send a request to the LLM embedding model with text
  */
  (cellSelection) => async (dispatch) => {
    dispatch({
      type: "request to embedding model started",
    });
    try {
      // Legal values are null, Array or TypedArray.  Null is initial state.
      if (!cellSelection) cellSelection = [];

      // These lines ensure that we convert any TypedArray to an Array.
      // This is necessary because JSON.stringify() does some very strange
      // things with TypedArrays (they are marshalled to JSON objects, rather
      // than being marshalled as a JSON array).
      cellSelection = Array.isArray(cellSelection)
        ? cellSelection
        : Array.from(cellSelection);

      const res = await fetch(
        `${globals.API.prefix}${globals.API.version}llmembs/obs`,
        {
          method: "POST",
          headers: new Headers({
            Accept: "application/json",
            "Content-Type": "application/json",
          }),
          body: JSON.stringify({
            cellSelection: { filter: { obs: { index: cellSelection } } },
          }),
          credentials: "include",
        }
      );

      if (!res.ok || res.headers.get("Content-Type") !== "application/json") {
        return dispatch({
          type: "request llm embeddings error",
          error: new Error(
            `Unexpected response ${res.status} ${
              res.statusText
            } ${res.headers.get("Content-Type")}}`
          ),
        });
      }

      const response = await res.json();
      return dispatch({
        type: "embedding model text response from cells",
        data: response,
      });
    } catch (error) {
      return dispatch({
        type: "request llm embeddings error",
        error,
      });
    }
  };

export const requestEmbeddingLLMWithText =
  /*
    Send a request to the LLM embedding model with text
  */
  (text) => async (dispatch) => {
    dispatch({
      type: "request to embedding model started",
    });
    try {
      const res = await fetch(
        `${globals.API.prefix}${globals.API.version}llmembs/text`,
        {
          method: "POST",
          headers: new Headers({
            Accept: "application/octet-stream",
            "Content-Type": "application/json",
          }),
          body: JSON.stringify({
            text,
          }),
          credentials: "include",
        }
      );

      if (
        !res.ok ||
        res.headers.get("Content-Type") !== "application/octet-stream"
      ) {
        return dispatch({
          type: "request llm embeddings error",
          error: new Error(
            `Unexpected response ${res.status} ${
              res.statusText
            } ${res.headers.get("Content-Type")}}`
          ),
        });
      }

      const buffer = await res.arrayBuffer();
      const dataframe = matrixFBSToDataframe(buffer);
      const col = dataframe.__columns[0];

      const annotationName = dataframe.colIndex.getLabel(0);

      dispatch({
        type: "embedding model annotation response from text",
      });

      return dispatch(annotationCreateContinuousAction(annotationName, col));
    } catch (error) {
      return dispatch({
        type: "request llm embeddings error",
        error,
      });
    }
  };


/*
  Action creator to interact with the http_bot endpoint
*/
export const startChatRequest = (messages, prompt, cellSelection) => async (dispatch) => {
  let newMessages = messages.concat({from: "human", value: prompt});
  dispatch({ type: "chat request start", newMessages });

  try {
    if (!cellSelection) cellSelection = [];

    // These lines ensure that we convert any TypedArray to an Array.
    // This is necessary because JSON.stringify() does some very strange
    // things with TypedArrays (they are marshalled to JSON objects, rather
    // than being marshalled as a JSON array).
    cellSelection = Array.isArray(cellSelection)
      ? cellSelection
      : Array.from(cellSelection);

    const pload = {
      messages: newMessages,  // TODO might need to add <image> to first message
      cellSelection: { filter: { obs: { index: cellSelection } } },
    };

    const response = await fetch(`${globals.API.prefix}${globals.API.version}llmembs/chat`, {
      method: 'POST',
      headers: new Headers({
        // Accept: "application/json",
        'Content-Type': 'application/json',
      }),
      body: JSON.stringify(pload),
    });

    if (!response.ok) {
      throw new Error('Failed to get response from the model');
    }

    // NOTE: The canonical way to solve this would probably be to use EventStreams. But it should also be possible with fetch as below
    // Stream the response (assuming the API sends back chunked responses)
    const reader = response.body.getReader();
    let chunksAll = new Uint8Array(0);
    let receivedLength = 0; // length at the moment
    while(true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      let temp = new Uint8Array(receivedLength + value.length);
      temp.set(chunksAll, 0); // copy the old data
      temp.set(value, receivedLength); // append the new chunk
      chunksAll = temp; // reassign the extended array
      receivedLength += value.length;

      // get the last chunk

      // Assuming chunksAll is the Uint8Array containing the data
      let lastZeroIndex = chunksAll.lastIndexOf(0);

      if (lastZeroIndex == -1) {
        continue;
      }
      let secondLastZeroIndex = chunksAll.lastIndexOf(0, lastZeroIndex - 1);
      // if secondLastZeroIndex is -1 (only 1 zero), go from the start
      let lastChunk = chunksAll.slice(secondLastZeroIndex+1, lastZeroIndex);

      // Decode into a string
      let result = new TextDecoder("utf-8").decode(lastChunk);

      // Parse the JSON (assuming the final string is a JSON object)
      const data = JSON.parse(result);

      // trim away the '<image>' string:
      data.text = data.text.replace("<image>", "");

      dispatch({ type: "chat request success", payload: data.text });
    }

  } catch (error) {
    dispatch({ type: "chat request failure", payload: error.message });
  }
};

/*
  Action creator to get score gene contributions
  NOTE: code is very similar to the other requests and could be DRYed
*/
export const geneContributionRequest =
  /*
    Send a request to the LLM embedding model with text
  */
  (text, cellSelection) => async (dispatch) => {
    dispatch({
      type: "request to embedding model started",
    });
    try {
      // Legal values are null, Array or TypedArray.  Null is initial state.
      if (!cellSelection) cellSelection = [];

      // These lines ensure that we convert any TypedArray to an Array.
      // This is necessary because JSON.stringify() does some very strange
      // things with TypedArrays (they are marshalled to JSON objects, rather
      // than being marshalled as a JSON array).
      cellSelection = Array.isArray(cellSelection)
        ? cellSelection
        : Array.from(cellSelection);

      const res = await fetch(
        `${globals.API.prefix}${globals.API.version}llmembs/interpret`,
        {
          method: "POST",
          headers: new Headers({
            Accept: "application/json",
            "Content-Type": "application/json",
          }),
          body: JSON.stringify({
            cellSelection: { filter: { obs: { index: cellSelection } } },
            text
          }),
          credentials: "include",
        }
      );

      if (!res.ok || res.headers.get("Content-Type") !== "application/json") {
        return dispatch({
          type: "request llm embeddings error",
          error: new Error(
            `Unexpected response ${res.status} ${
              res.statusText
            } ${res.headers.get("Content-Type")}}`
          ),
        });
      }

      const response = await res.json();
      return dispatch({
        type: "embedding model gene contributions response",
        data: JSON.stringify(response, null, 2),
      });
    } catch (error) {
      return dispatch({
        type: "request llm embeddings error",
        error,
      });
    }
  };
