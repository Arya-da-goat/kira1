// All neural weights and downloaded documents remain on this device. No backend.
export const DB_NAME = 'kira-autonomous-wikipedia-v1';
const VERSION = 1;
let opening;
export function openDB() {
  if (opening) return opening;
  opening = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      const articles = db.createObjectStore('articles', {keyPath: 'id'});
      articles.createIndex('subject', 'subject');
      const samples = db.createObjectStore('samples', {keyPath: 'id'});
      samples.createIndex('subject', 'subject');
      db.createObjectStore('meta', {keyPath: 'key'});
      db.createObjectStore('checkpoints', {keyPath: 'key'});
      db.createObjectStore('activity', {keyPath: 'id', autoIncrement: true});
    };
    req.onsuccess = () => { const db = req.result; db.onversionchange = () => db.close(); resolve(db); };
    req.onerror = () => { opening = null; reject(req.error); };
    req.onblocked = () => reject(new Error('Kira database is blocked by another tab. Close other Kira tabs.'));
  });
  return opening;
}
export async function read(store, key) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = db.transaction(store, 'readonly').objectStore(store).get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
export async function write(store, value) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readwrite');
    tx.objectStore(store).put(value);
    tx.oncomplete = () => resolve(value);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}
export async function all(store, count = 1000) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = db.transaction(store, 'readonly').objectStore(store).getAll(undefined, count);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
export async function keys(store, count = 50000) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = db.transaction(store, 'readonly').objectStore(store).getAllKeys(undefined, count);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
export async function count(store) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = db.transaction(store, 'readonly').objectStore(store).count();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
export async function getMeta(key, fallback = null) { return (await read('meta', key))?.value ?? fallback; }
export async function setMeta(key, value) { return write('meta', {key, value}); }
let activityWrites=0;
export async function addActivity(message, type = 'info') {
  const db = await openDB();
  if(++activityWrites%45===0){
    try{
      const total=await count('activity');
      if(total>700){
        await new Promise((resolve,reject)=>{
          const tx=db.transaction('activity','readwrite');
          let remaining=total-600;
          const cursor=tx.objectStore('activity').openCursor();
          cursor.onsuccess=()=>{const row=cursor.result;if(row&&remaining-->0){row.delete();row.continue();}};
          tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);
        });
      }
    }catch{/* Activity retention must not interrupt training. */}
  }
  return new Promise((resolve, reject) => {
    const tx = db.transaction('activity', 'readwrite');
    tx.objectStore('activity').add({at: Date.now(), message: String(message).slice(0, 240), type});
    tx.oncomplete = resolve; tx.onerror = () => reject(tx.error);
  });
}
export async function activities(limit = 35) { return (await all('activity', 2500)).slice(-limit).reverse(); }
