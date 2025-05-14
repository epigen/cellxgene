import React from "react";
import { connect } from "react-redux";
import { Button, InputGroup } from "@blueprintjs/core";
import actions from "../../actions";

const INITIAL_TEMPERATURE = 0.0;
const REGENERATE_TEMPERATURE = 1.0;
const SEARCH_KEYWORD_REGEX = /^\s*(show me|show|search for|search|find|which of these)( all)?( (samples|cells) (from|of|that|which|are))?:?\s*/i;  // important to have the longer ones before the shorter overlapping ones

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
  enableGeneScoreContributions: state.config?.parameters?.["enable-llmembs_gene_score_contribution"] ?? false,
}))

class ChatSideBar extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      inputText: "",
      conversationSample: null,
      likedMessages: [],
    };
    this.messagesEndRef = React.createRef(); // Create a ref for the messages container
    this.inputRef = React.createRef();
  }

  handleInputChange = (e) => {
    this.setState({ inputText: e.target.value });
  };

  buttonDisabled = () => {
    const { inputText, conversationSample } = this.state;
    const { obsCrossfilter, loading } = this.props;
    const countSelected = obsCrossfilter.countSelected();
    const totalObs = obsCrossfilter.annoMatrix.nObs;

    const onlyKeywordRegex = new RegExp(SEARCH_KEYWORD_REGEX.source + "$", "i");
    return (
      (inputText === "" && (countSelected === 0 || countSelected === totalObs)) ||
      (inputText.startsWith("/interpret") && !obsCrossfilter.countSelected()) ||
      (onlyKeywordRegex.test(inputText)) ||
      loading
    );
  };

  inputSubmit = () => {
    const { dispatch, obsCrossfilter, messages, enableGeneScoreContributions } = this.props;
    const { inputText, conversationSample } = this.state;
    const countSelected = obsCrossfilter.countSelected();
    const totalObs = obsCrossfilter.annoMatrix.nObs;

    if (this.buttonDisabled()) {
      return;
    }

    let submitMessages = messages;

    // Test if conversationSample changed or whether the chat is empty
    if (!Array.isArray(messages) || JSON.stringify(conversationSample) !== JSON.stringify(obsCrossfilter.allSelectedLabels()) || inputText === "") {
      submitMessages = [];
      this.setState({ conversationSample: obsCrossfilter.allSelectedLabels(), likedMessages: [] });
    }

    if (SEARCH_KEYWORD_REGEX.test(inputText) || countSelected === totalObs || countSelected === 0) {
      const search = inputText.replace(SEARCH_KEYWORD_REGEX, "");
      dispatch(actions.requestEmbeddingLLMWithText(search));

      let chatMessage = inputText;
      if (!SEARCH_KEYWORD_REGEX.test(inputText))
        chatMessage = `(Show me) ${inputText}`;


      let newMessages = submitMessages.concat({from: "human", value: chatMessage });
      dispatch({ type: "chat request start", newMessages });  // slight abuse, since we didn't start a real chat, but fair since it is all handled in here
    } else if (inputText.startsWith("/interpret") && enableGeneScoreContributions) {
      dispatch(actions.geneContributionRequest(inputText, obsCrossfilter.allSelectedLabels()));
    } else {
      // Dispatch the action to send the message

      dispatch(actions.startChatRequest(submitMessages, inputText || "Describe these cells in detail.", obsCrossfilter.allSelectedLabels(), INITIAL_TEMPERATURE));
    }
    this.setState({ inputText: "" }); // Clear the input after sending
  };

  handleEdit = (messageId) => {
    // This modifies the messages history and sets the 
    const { dispatch, messages } = this.props;
    const message = messages[messageId];
    const messagesSlice = messages.slice(0, messageId);

    dispatch(actions.resetChat(messagesSlice));
    this.setState({ inputText: message.value }); // Set the input box to the selected message for editing

    // focus the input 
    if (this.inputRef.current) {
      this.inputRef.current.focus();
    }
  }

  handleThumb = (messageId, thumbDirection) => {
    const { dispatch, messages } = this.props;
    const { inputText, conversationSample } = this.state;

    // copy messages array and take all messages up to (including) messageId
    const messagesCopy = messages.slice(0, messageId + 1);
    dispatch(actions.chatFeedback(messagesCopy, conversationSample, thumbDirection));

    // Regenerate response on negative feedback
    // one caveat is that the user might have changed the selection, but we ignore that for now
    if (thumbDirection === "down") {
      let submitMessages = messagesCopy.slice(0, -1);
      const userMessage = submitMessages.pop()["value"];
      dispatch(actions.startChatRequest(submitMessages, userMessage, conversationSample, REGENERATE_TEMPERATURE));
      // get rid of all liked messages with smaller index
      this.setState({ likedMessages: this.state.likedMessages.filter((likedMessageId) => likedMessageId > messageId) });
    } else {
      this.setState({ likedMessages: [...this.state.likedMessages, messageId] });
    }
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
  buttonText = () => {
    const { inputText, conversationSample } = this.state;
    const { obsCrossfilter, enableGeneScoreContributions } = this.props;
    const countSelected = obsCrossfilter.countSelected();
    const totalObs = obsCrossfilter.annoMatrix.nObs;

    let action;
    if (SEARCH_KEYWORD_REGEX.test(inputText) || countSelected === totalObs || countSelected === 0) {
      action = "Search for cells";
    } else {
      if (inputText.startsWith("/interpret") && enableGeneScoreContributions) {
        action = "Interpret";
      }
      else if (inputText == "") {
        action = "Describe";
      }
      else {
        const isNewSelection = JSON.stringify(conversationSample) !== JSON.stringify(obsCrossfilter.allSelectedLabels());
        action = isNewSelection ? "Chat about" : "Continue chat about";
      }

      const selectionLabel = countSelected === totalObs ? "all " + countSelected : "n=" + countSelected;
      action = action + " selected cells (mean of " + selectionLabel + " cells)";
    }
    return action;
  };

  renderMessages() {
    const { messages } = this.props;
    const { likedMessages } = this.state;
    return (
      <div
      >
        {messages.map((message, index) => (
          <div
            key={index}
            style={{
              display: 'flex',
              justifyContent: message.from === "human" ? 'flex-end' : 'flex-start',
              margin: "5px",
            }}
          >
            <div
              style={{
                display: 'flex',
                backgroundColor: message.from === "human" ? "#96c03a" : "#bee3ef",
                padding: "5px 10px",
                borderRadius: "5px",
                width: "90%", // Prevents the bubble from stretching too wide
                justifyContent: "space-between",
              }}
            >
              <span> {message.value} </span>
              { message.from === "gpt" ?
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  flexDirection: 'column'
                }}>
                  <button
                    style={{
                      border: "none",
                      background: "transparent",
                      cursor: likedMessages.includes(index) ? "default" : "pointer",
                      opacity: likedMessages.includes(index) ? 1 : 0.7,
                      transition: "opacity 0.3s ease",
                      padding: "3px 0px",
                      fontSize: "12pt"
                    }}
                    onClick={likedMessages.includes(index) ? null : () => this.handleThumb(index, "up")}
                    onMouseOver={likedMessages.includes(index) ? null : (e) => e.currentTarget.style.opacity = 1}
                    onMouseOut={likedMessages.includes(index) ? null : (e) => e.currentTarget.style.opacity = 0.7}
                  >
                    👍
                  </button>
                  <button
                    style={{
                      border: "none",
                      background: "transparent",
                      cursor: "pointer",
                      opacity: 0.7,
                      transition: "opacity 0.3s ease",
                      padding: "3px",
                      fontSize: "12pt"
                    }}
                    onClick={() => this.handleThumb(index, "down")}
                    onMouseOver={(e) => e.currentTarget.style.opacity = 1}
                    onMouseOut={(e) => e.currentTarget.style.opacity = 0.7}
                  >
                    👎
                  </button>
                </div>
                : <div>
                  <button
                    style={{
                      border: "none",
                      background: "transparent",
                      cursor: "pointer",
                      opacity: 0.7,
                      transition: "opacity 0.3s ease",
                      padding: "3px",
                      fontSize: "12pt"
                    }}
                    onClick={() => this.handleEdit(index)}
                    onMouseOver={(e) => e.currentTarget.style.opacity = 1}
                    onMouseOut={(e) => e.currentTarget.style.opacity = 0.7}
                    disabled={this.buttonDisabled()}
                  >
                    📝
                  </button>
                </div>
              }
            </div>
          </div>
        ))}
      </div>
    );
  }

  render() {
    const { messages, loading, obsCrossfilter, enableGeneScoreContributions } = this.props;
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
            whiteSpace: "pre-wrap",
            flex: 1,
          }}
          ref={this.messagesEndRef} // Attach the ref to this div
        >
          {typeof messages === "string" ? (
            messages
          ) : this.renderMessages()}
        </div>
        <div
          style={{
            margin: "5px 5px",
          }}
        >
          <InputGroup
            value={inputText}
            inputRef={this.inputRef}
            fill
            onChange={this.handleInputChange}
            placeholder="Type your request here and press <Enter>. For example: Show me T cells"
            onKeyDown={(e) => {
              if (e.key === "Enter" && !this.buttonDisabled()) {
                this.inputSubmit();
              }
            }}
          />
        </div>
        <div
          style={{
            margin: "5px 5px",
            display: "flex",
            flexDirection: "row",
            justifyContent: "right",
          }}
        >
          <Button
            onClick={this.inputSubmit}
            disabled={this.buttonDisabled()}
            loading={loading}
            style={{ margin: "0px 10px", padding: "0px 20px" }}
          >
            {this.buttonText()}
          </Button>
        </div>
      </div>
    );
  }
}

export default ChatSideBar;
