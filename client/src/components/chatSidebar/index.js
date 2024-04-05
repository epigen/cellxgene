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
      this.setState({ conversationSample: obsCrossfilter.allSelectedLabels(), likedMessages: [] });
    }
    dispatch(actions.startChatRequest(submitMessages, inputText, obsCrossfilter.allSelectedLabels()));
    this.setState({ inputText: "" }); // Clear the input after sending
  };

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
      dispatch(actions.startChatRequest(submitMessages, userMessage, conversationSample));
      // get rid of all liked messages with smaller index
      this.setState({ likedMessages: this.state.likedMessages.filter((likedMessageId) => likedMessageId > messageId) });
    } else {
      this.setState({ likedMessages: [...this.state.likedMessages, messageId] });
    }
  };

  geneContributionClicked = () => {
    const { dispatch, obsCrossfilter } = this.props;
    const { inputText } = this.state;

    dispatch(actions.geneContributionRequest(inputText, obsCrossfilter.allSelectedLabels()));

    // this.setState({ inputText: "" }); // Clear the input after sending
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
                alignItems: 'center',
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
                      opacity: likedMessages.includes(index) ? 1 : 0.5,
                      transition: "opacity 0.3s ease",
                      padding: "3px 0px"
                    }}
                    onClick={likedMessages.includes(index) ? null : () => this.handleThumb(index, "up")}
                    onMouseOver={likedMessages.includes(index) ? null : (e) => e.currentTarget.style.opacity = 1}
                    onMouseOut={likedMessages.includes(index) ? null : (e) => e.currentTarget.style.opacity = 0.5}
                  >
                    👍
                  </button>
                  <button
                    style={{
                      border: "none",
                      background: "transparent",
                      cursor: "pointer",
                      opacity: 0.5,
                      transition: "opacity 0.3s ease",
                      padding: "3px"
                    }}
                    onClick={() => this.handleThumb(index, "down")}
                    onMouseOver={(e) => e.currentTarget.style.opacity = 1}
                    onMouseOut={(e) => e.currentTarget.style.opacity = 0.5}
                  >
                    👎
                  </button>
                </div>
                : null
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
            fill
            onChange={this.handleInputChange}
            placeholder="E.g. Describe the sample of selected cells..."
            onKeyDown={(e) => {
              if (e.key === "Enter" && obsCrossfilter.countSelected() > 0 && inputText) {
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
            active={true}  // reflect that "enter" presses this one
            // autoFocus={true} // does not work, because focus needs to be on input field
            loading={loading}
            style={{ margin: "0px 0px", padding: "0px 20px" }}
          >
            Find cells
          </Button>
          { enableGeneScoreContributions &&
            <Button
              onClick={this.geneContributionClicked}
              disabled={
                obsCrossfilter.countSelected() === 0 ||
                  !inputText
              }
              loading={loading}
              style={{ margin: "0px 10px", padding: "0px 20px" }}
            >
              Interpret selected pseudocell (mean of n={obsCrossfilter.countSelected()})
            </Button>
          }
          <Button
            onClick={this.chatSelectedClick}
            disabled={
              obsCrossfilter.countSelected() === 0 ||
                !inputText
            }
            loading={loading}
            style={{ margin: "0px 10px", padding: "0px 20px" }}
          >
            {
              JSON.stringify(conversationSample) !== JSON.stringify(obsCrossfilter.allSelectedLabels()) ?
                "Start new conversation" : "Continue conversation"
            } about selected pseudocell (mean of {obsCrossfilter.countSelected() === obsCrossfilter.annoMatrix.nObs ? "all " + obsCrossfilter.countSelected() : "n=" + obsCrossfilter.countSelected()})
          </Button>
        </div>
      </div>
    );
  }
}

export default ChatSideBar;
