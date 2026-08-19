import React from 'react';
import {createRoot} from 'react-dom/client';

const callLegacy = (name, ...args) => {
  const handler = window[name];
  if (typeof handler === 'function') handler(...args);
};

function Brand() {
  return (
    <div className="brand">
      <div id="brandTitle">Pi BE-FAST</div>
    </div>
  );
}

function Header() {
  return (
    <header className="topbar">
      <Brand />
      <div className="top-actions">
        <label className="camera-picker">
          <span id="cameraLabel">摄像头</span>
          <select id="cameraSourceSelect" defaultValue="host" onChange={event => callLegacy('switchCameraSource', event.target.value)} aria-label="摄像头来源">
            <option id="hostCameraOption" value="host">主机摄像头</option>
            <option id="clientCameraOption" value="client">手机 / 当前设备</option>
          </select>
        </label>
        <button id="persistentHistoryButton" className="history-entry" onClick={() => callLegacy('openPersistentHistory')}>异常历史</button>
        <div id="languageSwitch" className="language" aria-label="语言">
          <button id="zhButton" className="active" onClick={() => callLegacy('setLanguage', 'zh')}>中文</button>
          <button id="enButton" onClick={() => callLegacy('setLanguage', 'en')}>EN</button>
        </div>
      </div>
    </header>
  );
}

const checks = [
  ['B', '平衡', '人工失衡观察 + 安全站立检测'],
  ['E', '眼动', '跟随左右目标，检查眼动范围'],
  ['F', '面部', '自然表情与微笑对称性'],
  ['A', '手臂', '双臂平举与单侧下落'],
  ['S', '言语', '长期自然语音监测或固定句朗读确认']
];

function BefastRail() {
  return (
    <nav id="steps" className="steps" aria-label="筛查进度">
      <span className="steps-label" aria-hidden="true">BE-FAST</span>
      {checks.map(([code, name]) => (
        <button key={code} className="step" data-code={code} onClick={() => callLegacy('startComponent', code)} aria-label={`${name}检测`}>{code}</button>
      ))}
    </nav>
  );
}

function ComponentGrid() {
  return checks.map(([code, name, description], index) => (
    <button key={code} className={`component-card ${name.toLowerCase()}-card`} onClick={() => callLegacy('startComponent', code)}>
      <strong>{code}</strong>
      <span id={`component${code}Title`}>{name}</span>
      <small id={`component${code}Text`}>{description}</small>
      <i aria-hidden="true">{String(index + 1).padStart(2, '0')}</i>
    </button>
  ));
}

function DataSidebar() {
  return (
    <aside id="dataSidebar" className="data-sidebar" aria-live="polite">
      <div className="factor-panel">
        <span id="factorKicker" className="factor-kicker">检测数据</span>
        <div className="factor-title-row">
          <h2 id="factorTitle">采集与判定</h2>
          <span id="factorVerdict" className="factor-verdict">等待项目</span>
        </div>
        <p id="factorSummary" className="factor-summary" hidden />
        <div id="collectionStrip" className="collection-strip" />
        <div id="factorSectionTitle" className="factor-section-title">关键指标</div>
        <div id="factorList" className="factor-list"><div className="factor-empty">等待数据</div></div>
        <div id="rawDataSectionTitle" className="factor-section-title">采集数据</div>
        <div id="rawDataList" className="raw-data-list" />
        <section id="fusionPanel" className="fusion-panel" aria-live="polite">
          <div id="fusionTitle" className="factor-section-title">融合观察</div>
          <p id="fusionNotice" className="fusion-notice">研究观察，不参与正式结论</p>
          <div id="fusionSummary" className="fusion-summary" />
          <div id="fusionDomainList" className="fusion-domain-list" />
        </section>
      </div>
    </aside>
  );
}

const mounts = [
  ['reactHeader', <Header />],
  ['reactSteps', <BefastRail />],
  ['reactComponentGrid', <ComponentGrid />],
  ['reactSidebar', <DataSidebar />]
];

for (const [id, tree] of mounts) {
  const target = document.getElementById(id);
  if (target) createRoot(target).render(tree);
}

requestAnimationFrame(() => {
  window.dispatchEvent(new Event('pibe-react-ready'));
});
