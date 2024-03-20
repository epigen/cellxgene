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
      conversationSample: null,
    };
    this.messagesEndRef = React.createRef(); // Create a ref for the messages container
  }

  handleInputChange = (e) => {
    this.setState({ inputText: e.target.value });
  };

  findCellsClick = () => {
    const { dispatch } = this.props;
    const { inputText } = this.state;
    dispatch(actions.requestEmbeddingLLMWithText(inputText));
  };

  chatSelectedClick = () => {
    const { dispatch, obsCrossfilter, messages } = this.props;
    const { inputText, conversationSample } = this.state;
    // Dispatch the action to send the message
    let submitMessages = messages;

    // Test if conversationSample changed
    if (JSON.stringify(conversationSample) !== JSON.stringify(obsCrossfilter.allSelectedLabels())) {
      submitMessages = [];
      this.setState({ conversationSample: obsCrossfilter.allSelectedLabels() });
    }
    dispatch(actions.startChatRequest(submitMessages, inputText, obsCrossfilter.allSelectedLabels()));
    this.setState({ inputText: "" }); // Clear the input after sending
  };

  componentDidUpdate(prevProps) {
    if (prevProps.messages !== this.props.messages) {
      this.scrollToBottom();
    }
  }

  scrollToBottom = () => {
    if (this.messagesEndRef.current) {
      this.messagesEndRef.current.scrollTop = this.messagesEndRef.current.scrollHeight;
    }
  };

  render() {
    const { messages, loading, obsCrossfilter } = this.props;
    const { inputText, conversationSample } = this.state;

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
          }}
          ref={this.messagesEndRef} // Attach the ref to this div
        >
          {typeof messages === "string" ? (
            messages
          ) : (
            <div
            >
              {messages.map((message) => (
                <div
                  style={message.from == "human" ? {
                    textAlign: "right",
                    backgroundColor: "#96c03a",
                    margin: "5px 5px",
                    padding: "5px 5px",
                    paddingLeft: "20px",
                    borderRadius: "5px",
                  } : {
                    textAlign: "left",
                    backgroundColor: "#bee3ef",
                    margin: "5px 5px",
                    padding: "5px 5px",
                    paddingRight: "20px",
                    borderRadius: "5px",
                  }}
                >
                  {message.value}
                </div>
              ))}
            </div>
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
            placeholder="E.g. Describe the sample of selected cells..."
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                this.chatSelectedClick();
              }
            }}
          />
        </div>
        <div
          style={{
            margin: "5px 5px",
            display: "flex",
            flexDirection: "row",
            justifyContent: "space-between",
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
            onClick={this.chatSelectedClick}
            disabled={
              obsCrossfilter.countSelected() === 0 ||
              obsCrossfilter.countSelected() === obsCrossfilter.annoMatrix.nObs ||
              !inputText
            }
            loading={loading}
            style={{ margin: "0px 10px", padding: "0px 20px" }}
          >
            {
              JSON.stringify(conversationSample) !== JSON.stringify(obsCrossfilter.allSelectedLabels()) ?
                "Start new conversation" : "Continue conversation"
            } about {obsCrossfilter.countSelected()} cells
          </Button>
        </div>
      </div>
    );
  }
}

export default ChatSideBar;
