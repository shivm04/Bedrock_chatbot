async function send(){

  const msg = document.getElementById("message").value;

  document.getElementById("chatbox")
    .innerHTML += `<p><b>You:</b> ${msg}</p>`;

  const res = await fetch(window.API_URL,{
      method:"POST",
      headers:{ "Content-Type":"application/json"},
      body: JSON.stringify({message:msg})
  });

  const data = await res.json();

  document.getElementById("chatbox")
    .innerHTML += `<p><b>Bot:</b> ${data.reply}</p>`;
}