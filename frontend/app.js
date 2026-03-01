async function send(){

  const input = document.getElementById("text");
  const message = input.value;

  if(!message) return;

  const messages = document.getElementById("messages");

  messages.innerHTML +=
    `<div class="msg user"><b>You:</b> ${message}</div>`;

  input.value="";

  const res = await fetch(window.API_URL,{
    method:"POST",
    headers:{ "Content-Type":"application/json"},
    body: JSON.stringify({ message })
  });

  const data = await res.json();

  messages.innerHTML +=
    `<div class="msg"><b>Claude:</b> ${data.reply}</div>`;

  messages.scrollTop = messages.scrollHeight;
}