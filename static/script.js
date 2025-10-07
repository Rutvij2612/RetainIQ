// Basic interactive dashboard script for RetainIQ
(function(){
  // Sections and buttons
  const sectionDataset = document.getElementById('sectionDataset');
  const sectionAdd = document.getElementById('sectionAdd');
  const sectionModify = document.getElementById('sectionModify');
  const sectionDelete = document.getElementById('sectionDelete');
  const sectionAnalysis = document.getElementById('sectionAnalysis');
  const sectionVisuals = document.getElementById('sectionVisuals');

  const btnViewDataset = document.getElementById('btnViewDataset');
  const btnAddRecord = document.getElementById('btnAddRecord');
  const btnModifyRecord = document.getElementById('btnModifyRecord');
  const btnDeleteRecord = document.getElementById('btnDeleteRecord');
  const btnViewAnalysis = document.getElementById('btnViewAnalysis');
  const btnVisualizeGraphs = document.getElementById('btnVisualizeGraphs');

  const table = document.getElementById('dataTable');
  const filterInput = document.getElementById('filterText');
  const refreshBtn = document.getElementById('refreshData');
  const visualsContainer = document.getElementById('visualsContainer');
  const analysisText = document.getElementById('analysisText');

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
    analysisText.innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;">
        <div><strong>Total</strong><div>${s.total_records}</div></div>
        <div><strong>Churn rate</strong><div>${(s.churn_rate*100).toFixed(1)}%</div></div>
        <div><strong>Churn counts</strong><div>${Object.entries(s.churn_counts||{}).map(([k,v])=>`${k}:${v}`).join(', ')}</div></div>
      </div>
      <h3>Feature Importance</h3>
      <pre style="white-space:pre-wrap;">${JSON.stringify(s.feature_importance||{}, null, 2)}</pre>
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
    if(v==='\n' || v==='' || v===null || v===undefined) return v;
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
      if(res.ok){ await fetchData(); await renderVisuals(); }
    }
    if(btn.dataset.action==='delete'){
      if(!confirm('Delete this record?')) return;
      const res = await fetch(`/api/data/${id}`, { method:'DELETE' });
      if(res.ok){ await fetchData(); await renderVisuals(); }
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
  refreshBtn?.addEventListener('click', async ()=>{ await fetchData(); await renderVisuals(); });

  // Dynamic forms for Add / Modify / Delete
  async function renderAddForm(){
    const container = sectionAdd;
    container.innerHTML = '<h2>Add New Record</h2><div id="addForm"></div>';
    const schema = await (await fetch('/api/schema')).json();
    const featureFields = schema.filter(f=>f.required);
    const form = document.createElement('form');
    form.className = 'form';
    featureFields.forEach(f=>{
      const label = document.createElement('label'); label.textContent = f.name;
      const input = f.choices ? buildSelect(f) : buildInput(f);
      form.appendChild(label); form.appendChild(input);
    });
    const submit = document.createElement('button'); submit.className='btn primary'; submit.type='submit'; submit.textContent='Add';
    form.appendChild(submit);
    form.addEventListener('submit', async (e)=>{
      e.preventDefault();
      const payload = collectForm(form);
      const res = await fetch('/api/data', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      if(res.ok){ await fetchData(); await renderVisuals(); alert('Record added'); }
      else { alert('Failed to add record'); }
    });
    container.appendChild(form);
  }

  function buildInput(f){
    const input = document.createElement('input');
    input.required = !!f.required;
    input.name = f.name;
    input.placeholder = f.name;
    input.type = f.type === 'number' ? 'number' : 'text';
    return input;
  }
  function buildSelect(f){
    const select = document.createElement('select');
    select.name = f.name; select.required = !!f.required;
    const blank = document.createElement('option'); blank.value=''; blank.textContent='-- select --'; select.appendChild(blank);
    (f.choices||[]).forEach(v=>{ const opt=document.createElement('option'); opt.value=v; opt.textContent=v; select.appendChild(opt); });
    return select;
  }
  function collectForm(form){
    const payload = {};
    new FormData(form).forEach((v,k)=>{ payload[k] = parseIfNumber(v); });
    return payload;
  }

  async function renderModifyForm(){
    const container = sectionModify;
    container.innerHTML = '<h2>Modify Existing Record</h2><div style="display:flex; gap:8px; align-items:end;"><div><label>RecordID</label><input id="modId" type="number" placeholder="RecordID"></div><button class="btn" id="loadMod">Load</button></div><div id="modFormWrap"></div>';
    document.getElementById('loadMod').onclick = ()=>loadForModify();
  }
  async function loadForModify(){
    const id = Number(document.getElementById('modId').value);
    if(!id) return;
    const rec = rows.find(r=>Number(r.RecordID)===id);
    if(!rec) { alert('Record not found'); return; }
    const schema = await (await fetch('/api/schema')).json();
    const fields = schema.filter(f=>f.name!=='RecordID');
    const wrap = document.getElementById('modFormWrap');
    wrap.innerHTML='';
    const form = document.createElement('form'); form.className='form';
    fields.forEach(f=>{
      const label = document.createElement('label'); label.textContent = f.name;
      const input = f.choices ? buildSelect(f) : buildInput(f);
      input.value = rec[f.name] ?? '';
      form.appendChild(label); form.appendChild(input);
    });
    const submit = document.createElement('button'); submit.className='btn primary'; submit.type='submit'; submit.textContent='Save';
    form.appendChild(submit);
    form.addEventListener('submit', async (e)=>{
      e.preventDefault();
      const payload = collectForm(form); delete payload.RecordID;
      const res = await fetch(`/api/data/${id}`, { method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      if(res.ok){ await fetchData(); await renderVisuals(); alert('Record updated'); }
      else { alert('Failed to update'); }
    });
    wrap.appendChild(form);
  }

  async function renderDeleteForm(){
    const container = sectionDelete;
    container.innerHTML = '<h2>Delete Record</h2><div style="display:flex; gap:8px; align-items:end;"><div><label>RecordID</label><input id="delId" type="number" placeholder="RecordID"></div><button class="btn" id="doDelete">Delete</button></div>';
    document.getElementById('doDelete').onclick = async ()=>{
      const id = Number(document.getElementById('delId').value);
      if(!id) return; if(!confirm('Delete this record?')) return;
      const res = await fetch(`/api/data/${id}`, { method:'DELETE' });
      if(res.ok){ await fetchData(); await renderVisuals(); alert('Deleted'); }
      else { alert('Record not found'); }
    };
  }

  // Interactive Visualizations: fetch URLs from backend and render images
  async function renderVisuals(){
    visualsContainer.innerHTML = '';
    const res = await fetch('/api/refresh_plots', { method:'POST' });
    const data = await res.json();
    const urls = data.plots || {};
    Object.values(urls).forEach(u=>{
      const img = document.createElement('img');
      img.src = `${u}?t=${Date.now()}`;
      visualsContainer.appendChild(img);
    });
  }

  function showOnly(section){
    [sectionDataset, sectionAdd, sectionModify, sectionDelete, sectionAnalysis, sectionVisuals].forEach(s=>{ if(s) s.style.display='none'; });
    if(section) section.style.display = '';
  }

  // Wire up action buttons
  btnViewDataset?.addEventListener('click', async ()=>{ showOnly(sectionDataset); if(rows.length===0) await fetchData(); });
  btnAddRecord?.addEventListener('click', async ()=>{ showOnly(sectionAdd); await renderAddForm(); });
  btnModifyRecord?.addEventListener('click', async ()=>{ showOnly(sectionModify); if(rows.length===0) await fetchData(); await renderModifyForm(); });
  btnDeleteRecord?.addEventListener('click', async ()=>{ showOnly(sectionDelete); await renderDeleteForm(); });
  btnViewAnalysis?.addEventListener('click', async ()=>{ showOnly(sectionAnalysis); await fetchStats(); });
  btnVisualizeGraphs?.addEventListener('click', async ()=>{ showOnly(sectionVisuals); await renderVisuals(); });

  // init
  // default shows only actions; sections remain hidden until clicked
})();

