/* Include external JavaScript files */
function include(file) {
  const script = document.createElement('script');
  script.src = file;
  script.type = 'text/javascript';
  script.defer = true;
  document.getElementsByTagName('head')[0].appendChild(script);
}

// Import additional components if needed
// include('./static/js/components/index.js');

// Bot pop-up intro
document.addEventListener("DOMContentLoaded", () => {
  const elemsTap = document.querySelector(".tap-target");
  const instancesTap = M.TapTarget.init(elemsTap, {});
  instancesTap.open();
  setTimeout(() => {
    instancesTap.close();
  }, 4000);
});

window.addEventListener('load', () => {
  $(document).ready(() => {
    // Bot pop-up intro
    $("div").removeClass("tap-target-origin");

    // Initialize dropdown for menu actions
    $(".dropdown-trigger").dropdown();

    // Toggle the chatbot screen
    $("#profile_div").click(() => {
      $(".profile_div").toggle();
      $(".widget").toggle();
    });

    // Clear chat contents
    $("#clear").click(() => {
      $(".chats").fadeOut("normal", () => {
        $(".chats").html("");
        $(".chats").fadeIn();
      });
    });

    // Close the chatbot widget
    $("#close").click(() => {
      $(".profile_div").toggle();
      $(".widget").toggle();
      scrollToBottomOfResults();
    });

    // Uncomment this if bot is configured to start conversation
    // customActionTrigger();
  });
});

/* Chat functionality */
$(document).ready(function () {
  const API_URL = "/chat"; // FastAPI backend endpoint

  // Append messages to the chat window
  function appendMessage(sender, message) {
    const avatar =
      sender === "user"
        ? '<img src="/static/img/usr.png" class="userAvatar" alt="User">'
        : '<img src="static/img/botAvatar.png" class="botAvatar" alt="Bot">';

    const messageHTML = `
      <div class="${sender}Msg">
        ${sender === "user" ? "" : avatar}
        <span>${message}</span>
        ${sender === "user" ? avatar : ""}
      </div>
    `;

    $("#chats").append(messageHTML);
    $("#chats").scrollTop($("#chats")[0].scrollHeight); // Auto-scroll
  }

  // Handle send button click
  $("#sendButton").click(function () {
    const userMessage = $("#userInput").val().trim();

    if (userMessage) {
      // Append user message to the chat
      appendMessage("user", userMessage);

      // Clear input field
      $("#userInput").val("");

      // Send the user message to the FastAPI backend
      $.ajax({
        url: API_URL,
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({ message: userMessage }),
        success: function (response) {
          const botResponse = response.response || "Sorry, I couldn't process your request.";
          appendMessage("bot", botResponse); // Append bot response
        },
        error: function () {
          appendMessage("bot", "There was an error communicating with the server. Please try again.");
        },
      });
    }
  });

  // Allow pressing 'Enter' to send a message
  $("#userInput").keypress(function (e) {
    if (e.which === 13 && !e.shiftKey) {
      e.preventDefault();
      $("#sendButton").click();
    }
  });
});
