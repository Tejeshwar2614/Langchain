const form = document.getElementById("chatForm");
const input = document.getElementById("question");
const messagesElement = document.getElementById("messages");
const chatScroll = document.getElementById("chatScroll");
const welcome = document.getElementById("welcome");
const clearButton = document.getElementById("clearChat");

const STORAGE_KEY = "clipwise-chat";
let messages = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");

function saveMessages() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
}

function renderMessage(role, content, extraClass = "") {
  const row = document.createElement("div");
  row.className = `message ${role} ${extraClass}`.trim();

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "You" : "✦";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = content;

  row.append(avatar, bubble);
  messagesElement.appendChild(row);
  chatScroll.scrollTop = chatScroll.scrollHeight;

  return row;
}

function refreshChat() {
  messagesElement.innerHTML = "";
  welcome.hidden = messages.length > 0;

  for (const message of messages) {
    renderMessage(message.role, message.content);
  }
}

async function sendQuestion(question) {
  const cleanQuestion = question.trim();
  if (!cleanQuestion) return;

  const previousMessages = [...messages];
  messages.push({ role: "user", content: cleanQuestion });
  saveMessages();
  refreshChat();

  input.value = "";
  input.style.height = "auto";

  const pending = renderMessage("assistant", "Thinking…", "pending");
  const sendButton = form.querySelector(".send-button");
  sendButton.disabled = true;

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: cleanQuestion,
        history: previousMessages,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "The assistant could not answer.");
    }

    const answer = data.answer;
    messages.push({ role: "assistant", content: answer });
    saveMessages();
    pending.remove();
    renderMessage("assistant", answer);
  } catch (error) {
    pending.remove();
    renderMessage("assistant", `Sorry, something went wrong: ${error.message}`);
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion(input.value);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 130)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    sendQuestion(button.dataset.prompt);
  });
});

clearButton.addEventListener("click", () => {
  messages = [];
  saveMessages();
  refreshChat();
});

refreshChat();