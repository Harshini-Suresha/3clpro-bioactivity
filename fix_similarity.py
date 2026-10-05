with open('/Users/harshinisuresha/Downloads/3clpro_project/site/index.html', 'r') as f:
    html = f.read()

old = '''/* ---- similarity (real Tanimoto) ---- */
const SM=D.sim,sb=$('smb');for(let k=0;k<NB;k++)sb.appendChild(document.createElement('i'));
$('fm').innerHTML='<table class="fm2"><thead><tr><th>#</th><th>Compound</th><th>IC50</th><th>Bits set</th></tr></thead><tbody>'+EX.map((m,i)=>`<tr data-r="${i}"><td>M${i+1}</td><td>${m.id}</td><td>${fmt(m.l)} nM</td><td>${m.on.length}</td></tr>`).join('')+'</tbody></table>';
let g='<div class="hm" style="grid-template-columns:auto repeat('+EX.length+',1fr)"><i></i>'+EX.map((_,i)=>`<span class="hc">M${i+1}</span>`).join('');
EX.forEach((_,i)=>{g+=`<span class="hr">M${i+1}</span>`;EX.forEach((_,j)=>{const v=SM[i][j];g+=`<button type="button" data-i="${i}" data-j="${j}" aria-label="M${i+1} and M${j+1}, similarity ${v.toFixed(2)}" style="background:rgba(0,146,125,${(.08+v*.82).toFixed(2)});color:${v>.5?'#fff':INK}">${v.toFixed(2)}</button>`})});
$('hm').innerHTML=g+'</div>';
let si=0,sj=Math.min(1,EX.length-1);
function sm(){
  const A=new Set(EX[si].on),B=new Set(EX[sj].on);let sh=0,oa=0,ob=0;
  [...sb.children].forEach(e=>e.className='');
  A.forEach(b=>{if(B.has(b)){sb.children[b].className='s';sh++}else{sb.children[b].className='a';oa++}});B.forEach(b=>{if(!A.has(b)){sb.children[b].className='c';ob++}});
  document.querySelectorAll('#fm tr[data-r]').forEach(r=>r.classList.toggle('sel',+r.dataset.r==si||+r.dataset.r==sj));
  document.querySelectorAll('#hm button').forEach(b=>b.classList.toggle('on',(+b.dataset.i==si&&+b.dataset.j==sj)||(+b.dataset.i==sj&&+b.dataset.j==si)));
  $('tan').textContent=SM[si][sj].toFixed(2);$('tanl').textContent=`M${si+1} and M${sj+1}`;
  $('smt').textContent=`${sh} shared bit${sh==1?'':'s'}, ${oa} only in M${si+1}, ${ob} only in M${sj+1}. ${sh} of the ${sh+oa+ob} bits set in either molecule are shared.`;
}
$('hm').addEventListener('click',e=>{const b=e.target.closest('button');if(b){si=+b.dataset.i;sj=+b.dataset.j;sm()}});
$('n_sim').textContent=`Real compounds, picked from the ${M.n_curated} curated molecules with a MaxMin diversity pick. Similarity is computed with RDKit on the same ${NB}-bit Morgan fingerprints used for modelling.`;
sm();'''

new = '''/* ---- similarity (real Tanimoto) ---- */
const SM=D.sim,sb=$('smb');for(let k=0;k<NB;k++)sb.appendChild(document.createElement('i'));
const onCnt={};EX.forEach(m=>m.on.forEach(b=>onCnt[b]=(onCnt[b]||0)+1));
$('fm').innerHTML='<table class="fm2"><thead><tr><th>#</th><th>Compound</th><th>IC50</th><th>pIC50</th><th>Bits set</th><th>Label</th></tr></thead><tbody>'+EX.map((m,i)=>{const L=lab(m.l);return `<tr data-r="${i}"><td>M${i+1}</td><td>${m.id}</td><td>${fmt(m.l)} nM</td><td>${(9-m.l).toFixed(1)}</td><td>${m.on.length}</td><td style="color:${LC[L]}">${L}</td></tr>`}).join('')+'</tbody></table>';
let g='<div class="hm" style="grid-template-columns:auto repeat('+EX.length+',1fr)"><i></i>'+EX.map((_,i)=>`<span class="hc">M${i+1}</span>`).join('');
EX.forEach((_,i)=>{g+=`<span class="hr">M${i+1}</span>`;EX.forEach((_,j)=>{const v=SM[i][j];g+=`<button type="button" data-i="${i}" data-j="${j}" aria-label="M${i+1} and M${j+1}, similarity ${v.toFixed(2)}" style="background:rgba(0,146,125,${(.08+v*.82).toFixed(2)});color:${v>.5?'#fff':INK}">${v.toFixed(2)}</button>`})});
$('hm').innerHTML=g+'</div>';
let si=0,sj=Math.min(1,EX.length-1);
function sm(){
  const A=new Set(EX[si].on),B=new Set(EX[sj].on);let sh=0,oa=0,ob=0,rareShared=0,rareA=0,rareB=0;
  [...sb.children].forEach(e=>e.className='');
  A.forEach(b=>{const cnt=onCnt[b]||0;const rare=cnt<=2;if(B.has(b)){sb.children[b].className='s';sh++;if(rare)rareShared++}else{sb.children[b].className='a';oa++;if(rare)rareA++}});B.forEach(b=>{const cnt=onCnt[b]||0;const rare=cnt<=2;if(!A.has(b)){sb.children[b].className='c';ob++;if(rare)rareB++}});
  document.querySelectorAll('#fm tr[data-r]').forEach(r=>r.classList.toggle('sel',+r.dataset.r==si||+r.dataset.r==sj));
  document.querySelectorAll('#hm button').forEach(b=>b.classList.toggle('on',(+b.dataset.i==si&&+b.dataset.j==sj)||(+b.dataset.i==sj&&+b.dataset.j==si)));
  $('tan').textContent=SM[si][sj].toFixed(2);$('tanl').textContent=`M${si+1} and M${sj+1}`;
  $('smt').textContent=`${sh} shared bit${sh==1?'':'s'}, ${oa} only in M${si+1}, ${ob} only in M${sj+1}. ${sh} of the ${sh+oa+ob} bits set in either molecule are shared.`;
  const Ainfo=EX[si],Binfo=EX[sj];const union=sh+oa+ob;
  $('smd').textContent=`Union: ${union} bits \u00b7 Jaccard: ${SM[si][sj].toFixed(3)} \u00b7 Rare bits (\u22642 examples): ${rareShared} shared, ${rareA} only A, ${rareB} only B \u00b7 M${si+1}: ${Ainfo.on.length} bits, M${sj+1}: ${Binfo.on.length} bits`;
}
$('hm').addEventListener('click',e=>{const b=e.target.closest('button');if(b){si=+b.dataset.i;sj=+b.dataset.j;sm()}});
$('n_sim').textContent=`Real compounds, picked from the ${M.n_curated} curated molecules with a MaxMin diversity pick. Similarity is computed with RDKit on the same ${NB}-bit Morgan fingerprints used for modelling. Coloured by Tanimoto: pale teal (low) to deep teal (high).`;
sm();'''

if old in html:
    html = html.replace(old, new)
    with open('/Users/harshinisuresha/Downloads/3clpro_project/site/index.html', 'w') as f:
        f.write(html)
    print('SUCCESS')
else:
    print('NOT FOUND')
    # Find the exact content
    idx = html.find('/* ---- similarity (real Tanimoto) ---- */')
    if idx >= 0:
        end_idx = html.find('/* ---- chemical space', idx)
        actual = html[idx:end_idx]
        print('Actual length:', len(actual))
        print('Expected length:', len(old))
        # Find first diff
        for i, (o, a) in enumerate(zip(old, actual)):
            if o != a:
                print(f'First diff at {i}: expected={repr(o)}, actual={repr(a)}')
                print(f'Context expected: {repr(old[max(0,i-30):i+30])}')
                print(f'Context actual:   {repr(actual[max(0,i-30):i+30])}')
                break