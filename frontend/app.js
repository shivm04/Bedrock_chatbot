async function send() {
  const input = document.getElementById("text");
  const message = input.value;

  if (!message) return;

  const messages = document.getElementById("messages");

  // 1. Add User Message
  messages.innerHTML += `
    <div class="msg user">
      <div class="avatar"><i class="fa-solid fa-user"></i></div>
      <div class="text"><b>You</b> ${message}</div>
    </div>`;

  input.value = "";

  // Scroll to bottom
  messages.scrollTop = messages.scrollHeight;

  // 2. Send to API
  try {
    const res = await fetch(window.API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });

    const data = await res.json();

    // 3. Add Bot Message
    messages.innerHTML += `
      <div class="msg bot">
        <div class="avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="text"><b>Claude</b> ${data.reply}</div>
      </div>`;
      
    messages.scrollTop = messages.scrollHeight;
  } catch (error) {
    console.error("Error:", error);
    messages.innerHTML += `<div class="msg bot" style="color: #ff4444;">Error connecting to AI.</div>`;
  }
}

// Allow pressing "Enter" to send
document.getElementById("text").addEventListener("keypress", function(event) {
  if (event.key === "Enter") {
    send();
  }
});