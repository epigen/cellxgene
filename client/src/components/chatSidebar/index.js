import React from "react";
import { connect } from "react-redux";
import { Button, InputGroup } from "@blueprintjs/core";
import actions from "../../actions";

function renderList(items) {
  return (
    <ul>
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

@connect((state) => ({
  ...state.llmEmbeddings,
  obsCrossfilter: state.obsCrossfilter,
}))
class ChatSideBar extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      inputText: "",
    };
  }

  handleInputChange = (e) => {
    this.setState({ inputText: e.target.value });
  };

  findCellsClick = () => {
    const { dispatch } = this.props;
    const { inputText } = this.state;
    dispatch(actions.requestEmbeddingLLMWithText(inputText));
  };

  // TODO abandon this soon
  describeSelectedClick = () => {
    const { dispatch, obsCrossfilter } = this.props;
    if (obsCrossfilter.allSelectedLabels()) {
      dispatch(
        actions.requestEmbeddingLLMWithCells(obsCrossfilter.allSelectedLabels())
      );
    }
  };

  chatSelectedClick = () => {
    const { dispatch, obsCrossfilter } = this.props;
    const { inputText } = this.state;
    // Dispatch the action to send the message
    dispatch(actions.startChatRequest(inputText, obsCrossfilter.allSelectedLabels()));
    this.setState({ inputText: "" }); // Clear the input after sending
  };

  render() {
    // TODO make sure that the button text is "<New?> Chat about <100> cells"
    const { outputText, loading, obsCrossfilter } = this.props;
    const { inputText } = this.state;

    return (
      <div
        style={{
          /* x y blur spread color */
          /* borderLeft: `1px solid ${globals.lightGrey}`, */
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: 0,
          flexGrow: 1,
          position: "relative",
          overflowY: "inherit",
          height: "inherit",
          width: "inherit",
        }}
      >
        <div
          style={{
            border: "1px solid #ccc",
            overflowY: "auto",
            padding: "5px 5px",
            margin: "5px 5px",
            flex: 1,
            whiteSpace: "pre",
          }}
        >
          {typeof outputText === "string" ? (
            outputText
          ) : (
            <ul>
              {outputText.map((item) => (
                <li key={item.library}>
                  <i>{item.library}</i>
                  {Array.isArray(item.keywords)
                    ? renderList(item.keywords)
                    : `: ${item.keywords}`}
                </li>
              ))}
            </ul>
          )}
        </div>
        <div
          style={{
            margin: "5px 5px",
          }}
        >
          <InputGroup
            value={inputText}
            fill
            onChange={this.handleInputChange}
            placeholder="Structural cells with immune function"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                this.findCellsClick();
              }
            }}
          />
        </div>
        <div
          style={{
            margin: "5px 5px",
            display: "flex",
            flexDirection: "row",
            justifyContent: "flex-end",
          }}
        >
          <Button
            onClick={this.findCellsClick}
            disabled={!inputText}
            loading={loading}
            style={{ margin: "0px 0px", padding: "0px 20px" }}
          >
            Find cells
          </Button>
          <Button
            onClick={this.describeSelectedClick}
            disabled={
              obsCrossfilter.countSelected() === 0 ||
              obsCrossfilter.countSelected() === obsCrossfilter.annoMatrix.nObs
            }
            loading={loading}
            style={{ margin: "0px 10px", padding: "0px 20px" }}
          >
            Describe selection
          </Button>
          <Button
            onClick={this.chatSelectedClick}
            disabled={
              obsCrossfilter.countSelected() === 0 ||
              obsCrossfilter.countSelected() === obsCrossfilter.annoMatrix.nObs
            }
            loading={loading}
            style={{ margin: "0px 10px", padding: "0px 20px" }}
          >
            Chat about selected
          </Button>
        </div>
      </div>
    );
  }
}

export default ChatSideBar;
