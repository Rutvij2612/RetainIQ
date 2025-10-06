// Basic interactive dashboard script for RetainIQ
(function(){
  const table = document.getElementById('dataTable');
  const filterInput = document.getElementById('filterText');
  const refreshBtn = document.getElementById('refreshData');
  const addBtn = document.getElementById('addRow');
  const statsDiv = document.getElementById('stats');

  let rows = [];
  let columns = [];
  let sortKey = null;
  let sortAsc = true;

  async function fetchData(){
    const res = await fetch('/api/data');
    rows = await res.json();
    columns = rows.length ? Object.keys(rows[0]) : [];
    renderTable();
    await fetchStats();
  }

  async function fetchStats(){
    const res = await fetch('/api/stats');
    const s = await res.json();
    statsDiv.innerHTML = `
      <div class="grid">
        <div><strong>Total</strong><div>${s.total_records}</div></div>
        <div><strong>Churn rate</strong><div>${(s.churn_rate*100).toFixed(1)}%</div></div>
      </div>
    `;
  }

  function filteredRows(){
    const q = (filterInput?.value || '').toLowerCase();
    if(!q) return [...rows];
    return rows.filter(r => String(r.City || '').toLowerCase().includes(q) || String(r['Zip Code'] || '').toLowerCase().includes(q));
  }

  function renderTable(){
    const data = filteredRows();
    if(sortKey){
      data.sort((a,b)=>{
        const av = a[sortKey];
        const bv = b[sortKey];
        if(av==bv) return 0; return (av>bv?1:-1)*(sortAsc?1:-1);
      });
    }
    const thead = `<thead><tr>${columns.map(c=>`<th data-key="${c}">${c}</th>`).join('')}<th>Actions</th></tr></thead>`;
    const tbody = `<tbody>${data.map(r=>`<tr>${columns.map(c=>`<td contenteditable="${c!=='RecordID'}" data-key="${c}" data-id="${r.RecordID}">${r[c]}</td>`).join('')}<td>
      <button class="btn" data-action="save" data-id="${r.RecordID}">Save</button>
      <button class="btn" data-action="delete" data-id="${r.RecordID}">Delete</button>
    </td></tr>`).join('')}</tbody>`;
    table.innerHTML = thead + tbody;
  }

  function collectRow(recordId){
    const tds = table.querySelectorAll(`td[data-id="${recordId}"]`);
    const obj = {};
    tds.forEach(td=>{ obj[td.dataset.key] = parseIfNumber(td.textContent.trim()); });
    return obj;
  }

  function parseIfNumber(v){
    if(v==='' || v===null || v===undefined) return v;
    const n = Number(v);
    return Number.isNaN(n) ? v : n;
  }

  table?.addEventListener('click', async (e)=>{
    const btn = e.target.closest('button');
    if(!btn) return;
    const id = Number(btn.dataset.id);
    if(btn.dataset.action==='save'){
      const payload = collectRow(id);
      delete payload.RecordID;
      const res = await fetch(`/api/data/${id}`, { method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      if(res.ok){ await fetchData(); await refreshPlots(); }
    }
    if(btn.dataset.action==='delete'){
      if(!confirm('Delete this record?')) return;
      const res = await fetch(`/api/data/${id}`, { method:'DELETE' });
      if(res.ok){ await fetchData(); await refreshPlots(); }
    }
  });

  table?.addEventListener('click', (e)=>{
    const th = e.target.closest('th');
    if(!th) return;
    const key = th.dataset.key;
    if(!key) return;
    if(sortKey===key) sortAsc = !sortAsc; else { sortKey = key; sortAsc = true; }
    renderTable();
  });

  filterInput?.addEventListener('input', ()=> renderTable());
  refreshBtn?.addEventListener('click', async ()=>{ await fetchData(); await refreshPlots(); });

  addBtn?.addEventListener('click', async ()=>{
    const featureCols = columns.filter(c=>c!=='RecordID' && c!=='Churn Value');
    const payload = {};
    for(const c of featureCols){
      const v = prompt(`Enter value for ${c}`);
      if(v===null) return; // cancel
      payload[c] = parseIfNumber(v);
    }
    const res = await fetch('/api/data', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    if(res.ok){ await fetchData(); await refreshPlots(); }
  });

  async function refreshPlots(){
    await fetch('/api/refresh_plots', { method:'POST' });
    // hard refresh plot images to bypass cache
    document.querySelectorAll('img').forEach(img=>{ if(img.src.includes('/static/')) img.src = img.src.split('?')[0] + `?t=${Date.now()}`; });
  }

  // init
  if(table){ fetchData(); }
})();

