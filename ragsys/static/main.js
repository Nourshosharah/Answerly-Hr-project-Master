/* core/static/main.js  – bare-minimum to show pages */
document.addEventListener('DOMContentLoaded', () => {
  /* mark first .page as active */
  const firstPage = document.querySelector('.page');
  if (firstPage) firstPage.classList.add('active');
});

const csrf = document.cookie.split('; ').find(r => r.startsWith('csrftoken')).split('=')[1];

async function sendMessage() {
  const msg   = document.getElementById('messageInput').value.trim();
  if (!msg) return;

  document.getElementById('sendBtnText').style.display = 'none';
  document.getElementById('sendSpinner').style.display = 'inline-block';

  const res = await fetch("{% url 'api_chat' %}", {
      method : 'POST',
      headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf},
      body   : JSON.stringify({
          message: msg,
          data_type: document.getElementById('ragMode').value,
          temperature: parseFloat(document.getElementById('tempSlider').value),
          show_thinking: document.getElementById('showThinking').checked
      })
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }

  // ---- append messages ----
  const box = document.getElementById('messagesList');
  box.insertAdjacentHTML('beforeend', data.user_html + data.bot_html);
  box.scrollTop = box.scrollHeight;

  // ---- clear input ----
  document.getElementById('messageInput').value = '';
  document.getElementById('sendBtnText').style.display = 'inline';
  document.getElementById('sendSpinner').style.display = 'none';
}

document.getElementById('messageForm').addEventListener('submit', e => {
  e.preventDefault();
  sendMessage();
});

/* ----------  show references  ---------- */
document.getElementById('refBtn').addEventListener('click', async () => {
  const panel = document.getElementById('refPanel');
  const content = document.getElementById('refContent');

  // toggle visibility
  if (!panel.classList.contains('hidden')) {
    panel.classList.add('hidden');
    return;
  }

  // fetch last references from the server
  const resp = await fetch("{% url 'api_refs' %}", {
    method : 'GET',
    headers: {'X-CSRFToken': csrf}
  });
  const data = await resp.json();
  if (data.error) { alert(data.error); return; }

  // render markdown (Tailwind prose makes it pretty)
  content.innerHTML = data.refs_html;
  panel.classList.remove('hidden');
  panel.scrollIntoView({behavior:'smooth'});
});