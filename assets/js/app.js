
        // Palette for custom diagrams (soft, desaturated tones adapting to theme)
        const ACCENT_PALETTE = [
            'var(--nv-green)',
            'var(--accent-blue)',
            'var(--accent-emerald)',
            'var(--accent-purple)',
            'var(--accent-amber)',
            'var(--accent-rose)'
        ];

        // ============================================================================
        // CUSTOM VISUAL DIAGRAM ENGINE (Redraws Mermaid into Modern HTML/CSS/SVG UI)
        // ============================================================================
        const CustomDiagramEngine = {
            diagramCounter: 0,
            rawStore: {},

            renderBlock(rawCode, customTitle = null) {
                const id = ++this.diagramCounter;
                const trimmed = rawCode.trim();
                this.rawStore[id] = trimmed;

                let diagramTypeLabel = "Sơ đồ Kiến trúc & Luồng dữ liệu";
                let customHtml = "";

                try {
                    if (trimmed.startsWith("mindmap")) {
                        diagramTypeLabel = "🧠 Bản đồ Tư duy Nhánh (Interactive Mindmap)";
                        customHtml = this.renderMindmap(trimmed);
                    } else if (trimmed.startsWith("sequenceDiagram")) {
                        diagramTypeLabel = "⏱️ Sơ đồ Tuần tự Thời gian (Sequence Timeline)";
                        customHtml = this.renderSequence(trimmed);
                    } else if (trimmed.startsWith("flowchart") || trimmed.startsWith("graph")) {
                        diagramTypeLabel = "⚡ Sơ đồ Kiến trúc & Luồng xử lý (Visual Architecture Flow)";
                        customHtml = this.renderFlowchart(trimmed);
                    } else {
                        customHtml = `<pre class="mermaid">${trimmed}</pre>`;
                    }
                } catch (e) {
                    console.warn("CustomDiagramEngine fallback:", e);
                    customHtml = `<pre class="mermaid">${trimmed}</pre>`;
                }

                const headerTitle = customTitle || diagramTypeLabel;

                return `
                    <div class="cd-wrapper" id="cd-wrap-${id}">
                        <div class="cd-header">
                            <div class="cd-title">
                                <span>✨ ${headerTitle}</span>
                            </div>
                            <div class="cd-mode-switch">
                                <button class="cd-mode-btn active" onclick="CustomDiagramEngine.switchMode(${id}, 'ui', this)">🎨 Giao diện Card Hiện đại</button>
                                <button class="cd-mode-btn" onclick="CustomDiagramEngine.switchMode(${id}, 'svg', this)">📐 Vector Graph (SVG)</button>
                            </div>
                        </div>
                        <div class="cd-canvas" id="cd-ui-${id}">
                            ${customHtml}
                        </div>
                        <div class="cd-canvas" id="cd-svg-${id}" style="display:none; text-align:center;"></div>
                    </div>
                `;
            },

            async switchMode(id, mode, btnEl) {
                const wrap = document.getElementById(`cd-wrap-${id}`);
                wrap.querySelectorAll('.cd-mode-btn').forEach(b => b.classList.remove('active'));
                btnEl.classList.add('active');

                const uiBox = document.getElementById(`cd-ui-${id}`);
                const svgBox = document.getElementById(`cd-svg-${id}`);

                if (mode === 'ui') {
                    uiBox.style.display = 'block';
                    svgBox.style.display = 'none';
                } else {
                    uiBox.style.display = 'none';
                    svgBox.style.display = 'block';
                    if (!svgBox.dataset.rendered && window.mermaid) {
                        svgBox.dataset.rendered = 'true';
                        const raw = this.rawStore[id];
                        try {
                            const { svg } = await mermaid.render(`mmd-svg-${id}-${Date.now()}`, raw);
                            svgBox.innerHTML = svg;
                        } catch (err) {
                            svgBox.innerHTML = `<pre style="text-align:left; color:var(--text-muted);">${raw}</pre>`;
                        }
                    }
                }
            },

            // 1. Render Mindmap Code into Node-Link Tree UI
            renderMindmap(code) {
                const lines = code.split(/\r?\n/).filter(l => l.trim() && !l.trim().startsWith('mindmap'));
                if (lines.length === 0) return '';

                let rootText = "NCA-GENL";
                const branches = [];
                let currentBranch = null;

                lines.forEach(line => {
                    const indent = line.search(/\S/);
                    const text = line.trim();
                    const rootMatch = text.match(/^root\(\((.+?)\)\)$/) || text.match(/^root\((.+?)\)$/);
                    if (rootMatch) {
                        rootText = rootMatch[1];
                    } else if (indent <= 4) {
                        currentBranch = { title: text, leaves: [] };
                        branches.push(currentBranch);
                    } else if (currentBranch) {
                        currentBranch.leaves.push(text);
                    }
                });

                return `
                    <div class="cm-mindmap-visual">
                        <div class="cm-root-hub">
                            <div style="font-size:1.5rem; margin-bottom:6px;">🎯</div>
                            <div>${rootText}</div>
                        </div>
                        <div class="cm-branches-col">
                            ${branches.map((b, i) => {
                                const color = ACCENT_PALETTE[i % ACCENT_PALETTE.length];
                                return `
                                    <div class="cm-branch-row" style="--branch-color: ${color};">
                                        <div class="cm-branch-node" style="--branch-color: ${color};">
                                            ${b.title}
                                        </div>
                                        <div class="cm-leaves-wrap" style="--branch-color: ${color};">
                                            ${b.leaves.map(leaf => `<span class="cm-leaf-pill">${leaf}</span>`).join('')}
                                        </div>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    </div>
                `;
            },

            // 2. Render Sequence Diagram Code into Interactive Timeline UI
            renderSequence(code) {
                const lines = code.split(/\r?\n/).map(l => l.trim()).filter(l => l && l !== 'sequenceDiagram' && l !== 'autonumber');
                const participants = [];
                const partMap = {};
                const events = [];
                let currentLoop = null;
                let stepNum = 0;

                lines.forEach(line => {
                    const pMatch = line.match(/^participant\s+([A-Za-z0-9_]+)\s+as\s+(.+)$/);
                    if (pMatch) {
                        const id = pMatch[1];
                        const label = pMatch[2].replace(/<br\s*\/?>/gi, ' ');
                        participants.push({ id, label });
                        partMap[id] = label;
                        return;
                    }
                    const noteMatch = line.match(/^Note\s+over\s+([^:]+):\s*(.+)$/i);
                    if (noteMatch) {
                        const target = currentLoop ? currentLoop.items : events;
                        target.push({ type: 'note', actors: noteMatch[1], text: noteMatch[2] });
                        return;
                    }
                    const loopMatch = line.match(/^loop\s+(.+)$/i);
                    if (loopMatch) {
                        currentLoop = { type: 'loop', title: loopMatch[1], items: [] };
                        events.push(currentLoop);
                        return;
                    }
                    if (line.toLowerCase() === 'end') {
                        currentLoop = null;
                        return;
                    }
                    const msgMatch = line.match(/^([A-Za-z0-9_]+)\s*(->>|-->>|->|-->)\s*([A-Za-z0-9_]+)\s*:\s*(.+)$/);
                    if (msgMatch) {
                        stepNum++;
                        const fromId = msgMatch[1];
                        const arrow = msgMatch[2];
                        const toId = msgMatch[3];
                        const text = msgMatch[4];
                        const target = currentLoop ? currentLoop.items : events;
                        target.push({
                            type: 'msg',
                            step: stepNum,
                            from: partMap[fromId] || fromId,
                            to: partMap[toId] || toId,
                            dashed: arrow.startsWith('--'),
                            text
                        });
                    }
                });

                const renderEventItem = (ev) => {
                    if (ev.type === 'note') {
                        return `<div class="cs-note-banner">📌 ${ev.text}</div>`;
                    }
                    if (ev.type === 'loop') {
                        return `
                            <div class="cs-loop-box">
                                <div class="cs-loop-label">🔁 VÒNG LẶP: ${ev.title}</div>
                                ${ev.items.map(renderEventItem).join('')}
                            </div>
                        `;
                    }
                    if (ev.type === 'msg') {
                        return `
                            <div class="cs-msg-row">
                                <span class="cs-msg-step">${ev.step}</span>
                                <span class="cs-msg-route">
                                    <span>${ev.from}</span>
                                    <span style="color:var(--nv-green-light)">${ev.dashed ? '⇢' : '➔'}</span>
                                    <span>${ev.to}</span>
                                </span>
                                <span class="cs-msg-text">${ev.text}</span>
                            </div>
                        `;
                    }
                    return '';
                };

                return `
                    <div class="cs-seq-table">
                        <div class="cs-participants" style="grid-template-columns: repeat(${Math.max(1, participants.length)}, 1fr);">
                            ${participants.map((p, idx) => `
                                <div class="cs-actor-box" style="border-color: ${ACCENT_PALETTE[idx % ACCENT_PALETTE.length]};">
                                    🖥️ ${p.label}
                                </div>
                            `).join('')}
                        </div>
                        <div>
                            ${events.map(renderEventItem).join('')}
                        </div>
                    </div>
                `;
            },

            // 3. Parse and Render Flowchart into Structured Architecture Cards & Pipelines
            renderFlowchart(code) {
                const rawLines = code.split(/\r?\n/);
                let mainDir = 'TD';
                const firstLineMatch = rawLines[0].trim().match(/^(?:flowchart|graph)\s+(TD|TB|LR|RL|BT)/i);
                if (firstLineMatch) {
                    mainDir = firstLineMatch[1].toUpperCase();
                }

                // Data structures
                const nodes = {}; // id -> { id, rawLabel, shape, subgraphId }
                const subgraphs = []; // { id, title, dir, parentId, nodeIds: [], childSubgraphs: [] }
                const sgMap = {};
                const edges = []; // { from, to, label, style }
                const sgStack = [];

                const registerNode = (id, rawLabel = null, shape = 'rect') => {
                    if (!nodes[id]) {
                        nodes[id] = {
                            id,
                            rawLabel: rawLabel || id,
                            shape,
                            subgraphId: sgStack.length > 0 ? sgStack[sgStack.length - 1].id : null
                        };
                        if (sgStack.length > 0) {
                            const currSg = sgStack[sgStack.length - 1];
                            if (!currSg.nodeIds.includes(id)) currSg.nodeIds.push(id);
                        }
                    } else if (rawLabel && nodes[id].rawLabel === id) {
                        nodes[id].rawLabel = rawLabel;
                        nodes[id].shape = shape;
                    }
                };

                // First join multiline quoted strings inside brackets if any
                const lines = [];
                let buffer = "";
                let quoteCount = 0;
                for (let i = 1; i < rawLines.length; i++) {
                    const l = rawLines[i].trim();
                    if (!l || l.startsWith('%%')) continue;
                    buffer += (buffer ? " " : "") + l;
                    quoteCount += (l.match(/"/g) || []).length;
                    if (quoteCount % 2 === 0) {
                        lines.push(buffer);
                        buffer = "";
                        quoteCount = 0;
                    }
                }
                if (buffer) lines.push(buffer);

                // Parse node token helper: e.g. A["Label"] or Q1{"Decision"} or VDB[("DB")] or Sum(("Circle"))
                const parseNodeToken = (tokenStr) => {
                    const t = tokenStr.trim();
                    let m;
                    if ((m = t.match(/^([A-Za-z0-9_]+)\s*\(\(\s*"?([\s\S]*?)"?\s*\)\)$/))) {
                        registerNode(m[1], m[2], 'circle');
                        return m[1];
                    }
                    if ((m = t.match(/^([A-Za-z0-9_]+)\s*\(\[\s*"?([\s\S]*?)"?\s*\]\)$/))) {
                        registerNode(m[1], m[2], 'stadium');
                        return m[1];
                    }
                    if ((m = t.match(/^([A-Za-z0-9_]+)\s*\[\(\s*"?([\s\S]*?)"?\s*\)\]$/))) {
                        registerNode(m[1], m[2], 'db');
                        return m[1];
                    }
                    if ((m = t.match(/^([A-Za-z0-9_]+)\s*\{\s*"?([\s\S]*?)"?\s*\}$/))) {
                        registerNode(m[1], m[2], 'diamond');
                        return m[1];
                    }
                    if ((m = t.match(/^([A-Za-z0-9_]+)\s*\[\s*"?([\s\S]*?)"?\s*\]$/))) {
                        registerNode(m[1], m[2], 'rect');
                        return m[1];
                    }
                    if ((m = t.match(/^([A-Za-z0-9_]+)$/))) {
                        if (!sgMap[m[1]]) registerNode(m[1]);
                        return m[1];
                    }
                    return t;
                };

                lines.forEach(line => {
                    // Check subgraph
                    const sgMatch = line.match(/^subgraph\s+([A-Za-z0-9_]+)(?:\s*\[\s*"?(.+?)"?\s*\])?$/);
                    if (sgMatch) {
                        const sgId = sgMatch[1];
                        const sgTitle = sgMatch[2] || sgId;
                        const parent = sgStack.length > 0 ? sgStack[sgStack.length - 1] : null;
                        const newSg = {
                            id: sgId,
                            title: sgTitle,
                            dir: 'TB',
                            parentId: parent ? parent.id : null,
                            nodeIds: [],
                            childSubgraphs: []
                        };
                        sgMap[sgId] = newSg;
                        if (parent) parent.childSubgraphs.push(newSg);
                        else subgraphs.push(newSg);
                        sgStack.push(newSg);
                        return;
                    }
                    if (line === 'end') {
                        sgStack.pop();
                        return;
                    }
                    const dirMatch = line.match(/^direction\s+(TB|TD|LR|RL|BT)/i);
                    if (dirMatch) {
                        if (sgStack.length > 0) sgStack[sgStack.length - 1].dir = dirMatch[1].toUpperCase();
                        return;
                    }

                    // Check if line has edges: supports -->, ==>, -.->, ---, <==>, <--> and labels |...| or -- "..." -->
                    const normalizedLine = line
                        .replace(/--\s*"([^"]+)"\s*-->/g, '-->|"$1"|')
                        .replace(/==\s*"([^"]+)"\s*==>/g, '==>|"$1"|')
                        .replace(/-\.\s*"([^"]+)"\s*\.->/g, '-.->|"$1"|');

                    const arrowRegex = /\s*(<==>|==>|<-->|-\.->|-->|---|~~~)\s*(?:\|"([^"]*)"\||\|([^|]*)\|)?\s*/g;
                    if (arrowRegex.test(normalizedLine)) {
                        arrowRegex.lastIndex = 0;
                        const parts = [];
                        const connectors = [];
                        let lastIdx = 0;
                        let match;
                        while ((match = arrowRegex.exec(normalizedLine)) !== null) {
                            parts.push(normalizedLine.substring(lastIdx, match.index));
                            connectors.push({
                                type: match[1],
                                label: (match[2] || match[3] || '').trim()
                            });
                            lastIdx = arrowRegex.lastIndex;
                        }
                        parts.push(normalizedLine.substring(lastIdx));

                        for (let c = 0; c < connectors.length; c++) {
                            const leftGroup = parts[c].split('&').map(s => parseNodeToken(s));
                            const rightGroup = parts[c + 1].split('&').map(s => parseNodeToken(s));
                            const conn = connectors[c];
                            if (conn.type === '~~~') continue; // invisible link
                            leftGroup.forEach(u => {
                                rightGroup.forEach(v => {
                                    edges.push({
                                        from: u,
                                        to: v,
                                        label: conn.label,
                                        dashed: conn.type === '-.->',
                                        thick: conn.type === '==>' || conn.type === '<==>',
                                        bidir: conn.type === '<==>' || conn.type === '<-->'
                                    });
                                });
                            });
                        }
                    } else {
                        parseNodeToken(line);
                    }
                });

                // Helper to render a single Node Card
                const renderNodeCard = (nodeId, colorIdx = 0) => {
                    const n = nodes[nodeId];
                    if (!n) return '';
                    const accent = ACCENT_PALETTE[colorIdx % ACCENT_PALETTE.length];
                    const parts = n.rawLabel.split(/<br\s*\/?>/i).map(s => s.trim()).filter(Boolean);
                    const rawTitle = parts[0] || n.id;
                    const bodyLines = parts.slice(1);

                    // Extract step prefix like "1. " or "Bước 1: "
                    let stepBadge = '';
                    let cleanTitle = rawTitle;
                    const stepMatch = rawTitle.match(/^((?:Bước|Giai đoạn|Cấp độ|Layer|Tầng|Pha)\s*\d+|\d+\.)\s*:?\s*(.+)$/i);
                    if (stepMatch) {
                        stepBadge = `<span class="cf-node-step-badge" style="--node-accent:${accent}">${stepMatch[1].replace(/\.$/, '')}</span>`;
                        cleanTitle = stepMatch[2];
                    }

                    return `
                        <div class="cf-node shape-${n.shape}" style="--node-accent: ${accent};">
                            <div class="cf-node-title">
                                ${stepBadge}
                                <span>${cleanTitle}</span>
                            </div>
                            ${bodyLines.length > 0 ? `
                                <ul class="cf-node-lines">
                                    ${bodyLines.map(line => {
                                        const cleaned = line.replace(/^[•\-*]\s*/, '');
                                        return `<li class="cf-node-line">
                                            <span class="cf-node-line-bullet" style="--node-accent:${accent}">▸</span>
                                            <span>${cleaned}</span>
                                        </li>`;
                                    }).join('')}
                                </ul>
                            ` : ''}
                        </div>
                    `;
                };

                // Helper to render a list of nodes arranged by topological rank or chain
                const renderNodeGroupFlow = (nodeIdList, dir = 'TD', baseColorIdx = 0) => {
                    if (nodeIdList.length === 0) return '';
                    const set = new Set(nodeIdList);
                    const localEdges = edges.filter(e => set.has(e.from) && set.has(e.to));

                    // Identify back-edges (cycles/feedback loops) using order in nodeIdList
                    const idxMap = {};
                    nodeIdList.forEach((id, i) => idxMap[id] = i);
                    const forwardEdges = [];
                    const feedbackEdges = [];
                    localEdges.forEach(e => {
                        if (idxMap[e.from] >= idxMap[e.to] && e.from !== e.to) {
                            feedbackEdges.push(e);
                        } else {
                            forwardEdges.push(e);
                        }
                    });

                    // Compute topological ranks
                    const inDegree = {};
                    const adj = {};
                    nodeIdList.forEach(id => { inDegree[id] = 0; adj[id] = []; });
                    forwardEdges.forEach(e => {
                        adj[e.from].push(e);
                        inDegree[e.to] = (inDegree[e.to] || 0) + 1;
                    });

                    const ranks = [];
                    const visited = new Set();
                    let currentRank = nodeIdList.filter(id => inDegree[id] === 0);
                    if (currentRank.length === 0) currentRank = [nodeIdList[0]];

                    while (currentRank.length > 0) {
                        ranks.push(currentRank);
                        currentRank.forEach(id => visited.add(id));
                        const nextRank = [];
                        currentRank.forEach(u => {
                            (adj[u] || []).forEach(edge => {
                                inDegree[edge.to]--;
                                if (inDegree[edge.to] <= 0 && !visited.has(edge.to) && !nextRank.includes(edge.to)) {
                                    nextRank.push(edge.to);
                                }
                            });
                        });
                        if (nextRank.length === 0 && visited.size < nodeIdList.length) {
                            const remaining = nodeIdList.find(id => !visited.has(id));
                            if (remaining) nextRank.push(remaining);
                        }
                        currentRank = nextRank;
                    }

                    const isHoriz = (dir === 'LR' || dir === 'RL') && ranks.length <= 4 && ranks.every(r => r.length === 1);
                    const arrowChar = isHoriz ? (dir === 'RL' ? '⬅' : '➔') : '⬇';

                    let html = `<div class="cf-ranks ${isHoriz ? 'dir-lr' : ''}">`;
                    ranks.forEach((rankNodes, rIdx) => {
                        // Incoming edge labels from previous rank to this rank
                        if (rIdx > 0) {
                            const prevSet = new Set(ranks[rIdx - 1]);
                            const currSet = new Set(rankNodes);
                            const connecting = forwardEdges.filter(e => prevSet.has(e.from) && currSet.has(e.to));
                            const labels = [...new Set(connecting.map(e => e.label).filter(Boolean))];
                            const isBidir = connecting.some(e => e.bidir);
                            html += `
                                <div class="cf-connector">
                                    ${labels.map(l => `<span class="cf-edge-pill">${l}</span>`).join('')}
                                    <span class="cf-arrow-svg">${isBidir ? '⬍' : arrowChar}</span>
                                </div>
                            `;
                        }

                        html += `<div class="cf-rank-group">`;
                        rankNodes.forEach((nid, nIdx) => {
                            // Check if this node has a specific incoming label from a decision or parent when multiple nodes in rank
                            let branchBadge = '';
                            if (rankNodes.length > 1 && rIdx > 0) {
                                const inc = forwardEdges.find(e => e.to === nid && e.label);
                                if (inc) {
                                    branchBadge = `<div style="margin-bottom:6px;"><span class="cf-edge-pill">↳ ${inc.label}</span></div>`;
                                }
                            }
                            html += `<div>${branchBadge}${renderNodeCard(nid, baseColorIdx)}</div>`;
                        });
                        html += `</div>`;
                    });
                    html += `</div>`;

                    if (feedbackEdges.length > 0) {
                        html += `<div class="cf-feedback-loops">`;
                        feedbackEdges.forEach(fe => {
                            const fromTitle = (nodes[fe.from]?.rawLabel || fe.from).split(/<br/i)[0];
                            const toTitle = (nodes[fe.to]?.rawLabel || fe.to).split(/<br/i)[0];
                            html += `
                                <div class="cf-feedback-item">
                                    <span>🔄 Nhánh vòng lặp / Đường nối tắt:</span>
                                    <strong>${fromTitle}</strong>
                                    <span>⇢</span>
                                    <strong>${toTitle}</strong>
                                    ${fe.label ? `<span class="cf-edge-pill" style="margin:0 0 0 auto;">${fe.label}</span>` : ''}
                                </div>
                            `;
                        });
                        html += `</div>`;
                    }

                    return html;
                };

                // SPECIAL CASE 1: Concentric Nested Subgraphs (e.g. AI > ML > DL > GenAI)
                if (subgraphs.length === 1 && subgraphs[0].childSubgraphs.length > 0) {
                    const renderConcentric = (sg, depth = 0) => {
                        const color = ACCENT_PALETTE[depth % ACCENT_PALETTE.length];
                        return `
                            <div class="cf-concentric-shell" style="--shell-color: ${color};">
                                <div class="cf-concentric-title">⭕ ${sg.title}</div>
                                ${sg.nodeIds.map((nid) => renderNodeCard(nid, depth)).join('')}
                                ${sg.childSubgraphs.map(child => renderConcentric(child, depth + 1)).join('')}
                            </div>
                        `;
                    };
                    return renderConcentric(subgraphs[0], 0);
                }

                // SPECIAL CASE 2: Parallel Bridge Mapping between 2 Subgraphs (e.g. CPU_World ==> GPU_World RAPIDS)
                if (subgraphs.length === 2 && subgraphs[0].nodeIds.length === subgraphs[1].nodeIds.length && subgraphs[0].nodeIds.length >= 3) {
                    const sgA = subgraphs[0];
                    const sgB = subgraphs[1];
                    const bridgeEdges = edges.filter(e => sgA.nodeIds.includes(e.from) && sgB.nodeIds.includes(e.to));
                    if (bridgeEdges.length >= 3) {
                        return `
                            <div class="cf-bridge-header">
                                <div class="cf-subgraph-header" style="--sg-color:${ACCENT_PALETTE[1]}; justify-content:center;">🐢 ${sgA.title}</div>
                                <div></div>
                                <div class="cf-subgraph-header" style="--sg-color:${ACCENT_PALETTE[0]}; justify-content:center;">🚀 ${sgB.title}</div>
                            </div>
                            <div class="cf-bridge-grid">
                                ${bridgeEdges.map((be, idx) => `
                                    <div class="cf-bridge-row">
                                        ${renderNodeCard(be.from, 1)}
                                        <div class="cf-bridge-arrow">
                                            <span>${be.label || 'Chuyển đổi'}</span>
                                            <span class="cf-bridge-arrow-icon">⟹</span>
                                        </div>
                                        ${renderNodeCard(be.to, 0)}
                                    </div>
                                `).join('')}
                            </div>
                        `;
                    }
                }

                // CASE 3: Diagram with Subgraphs (and optional top-level root nodes like ML --> Sup & Unsup & RL)
                if (subgraphs.length > 0) {
                    const topLevelNodes = Object.values(nodes).filter(n => !n.subgraphId).map(n => n.id);
                    const renderSubgraphBox = (sg, idx) => {
                        const color = ACCENT_PALETTE[idx % ACCENT_PALETTE.length];
                        return `
                            <div class="cf-subgraph" style="--sg-color: ${color};">
                                <div class="cf-subgraph-header" style="--sg-color: ${color};">
                                    <span>📦 ${sg.title}</span>
                                </div>
                                ${sg.childSubgraphs.length > 0 ? `
                                    <div class="cf-container dir-lr" style="margin-bottom:12px;">
                                        ${sg.childSubgraphs.map((csg, cIdx) => renderSubgraphBox(csg, idx + cIdx + 1)).join('')}
                                    </div>
                                ` : ''}
                                ${renderNodeGroupFlow(sg.nodeIds, sg.dir, idx)}
                            </div>
                        `;
                    };

                    // Check if subgraphs themselves are connected in sequence (e.g., Infra ==> CoreAI ==> Lifecycle ==> Prod)
                    const sgChainEdges = edges.filter(e => sgMap[e.from] && sgMap[e.to]);

                    let html = `<div class="cf-container">`;
                    if (topLevelNodes.length > 0) {
                        html += renderNodeGroupFlow(topLevelNodes, mainDir, 0);
                        html += `<div class="cf-connector"><span class="cf-arrow-svg">⬇</span></div>`;
                    }

                    const useGridForSubgraphs = sgChainEdges.length === 0 && subgraphs.length >= 2 && subgraphs.length <= 3;
                    html += `<div class="${useGridForSubgraphs ? 'cf-container dir-lr' : 'cf-container'}">`;
                    subgraphs.forEach((sg, idx) => {
                        if (idx > 0 && sgChainEdges.length > 0) {
                            html += `<div class="cf-connector"><span class="cf-arrow-svg">⬇</span></div>`;
                        }
                        html += renderSubgraphBox(sg, idx);
                    });
                    html += `</div></div>`;
                    return html;
                }

                // CASE 4: Pure Node Flowchart (No Subgraphs)
                const allNodeIds = Object.keys(nodes);
                return renderNodeGroupFlow(allNodeIds, mainDir, 0);
            }
        };

        // Structured Per-Module Visual Mindmap Summaries
        const MODULE_MINDMAPS = {
            readme: {
                root: "🎓 Bản Đồ Tổng Quan Chứng Chỉ NVIDIA NCA-GENL (50 / 50–60 Câu trong 60 Phút)",
                mermaid: `mindmap
  root((NVIDIA NCA-GENL))
    1. Core ML & AI 30%
      Module 01: AI Infrastructure
      Module 02: AI/ML Fundamentals
      Module 03: GenAI & LLMs
      Module 04: Transformer Arch
      Module 10: DL & Diffusion
    2. Software Dev 24%
      Module 06: RAG & Python APIs
      Module 08: NeMo, Triton, NIM, TensorRT-LLM
      Hugging Face & ONNX
    3. Experimentation 22%
      Module 05: BLEU, ROUGE, F1, PPL
      Module 06: Prompting, LoRA/QLoRA
      Module 11: Hypothesis, A/B, Ablation
    4. Data Analysis & Vis 14%
      Module 07 & 11: EDA, IQR, Charts, Leakage
      Module 08: RAPIDS cuDF, cuML, cuGraph
    5. Trustworthy AI 10%
      Module 09: Guardrails Colang
      Confidential Computing TEE
      XAI SHAP/LIME & Garak`,
                branches: [
                    { title: "📌 Cấu trúc Đề thi", items: ["<strong>50 câu</strong> (bảng tổng quan) / <strong>50–60 câu</strong> (mô tả chi tiết) trong <strong>60 phút</strong>", "<strong>Không trừ điểm câu sai</strong> -> Không bỏ trống bất kỳ câu nào", "Thi Online Proctored ($125 USD), tiếng Anh, hiệu lực 2 năm"] },
                    { title: "🔥 3 Miền Trọng Tâm Nhất (76%)", items: ["<strong>Core ML & AI (30%):</strong> Hạ tầng GPU, ML/DL, Transformer, Diffusion", "<strong>Software Dev (24%):</strong> Python API, RAG, NeMo, Triton, TensorRT-LLM, NIM", "<strong>Experimentation (22%):</strong> Thiết kế thực nghiệm, Prompting, LoRA/QLoRA, Metrics"] },
                    { title: "⚡ 2 Miền Ăn Điểm Nhanh (24%)", items: ["<strong>Data Analysis & Vis (14%):</strong> EDA, Biểu đồ, Data Leakage, Imbalanced Data, RAPIDS", "<strong>Trustworthy AI (10%):</strong> Hallucination, Bias, NeMo Guardrails, Confidential Computing"] }
                ]
            },
            mod01: {
                root: "🖥️ Mindmap Module 01: AI Infrastructure (Hạ Tầng AI Chuyên Dụng)",
                mermaid: `mindmap
  root((Module 01: AI Infra))
    Data Center & Stack
      East-West Traffic
      Liquid Cooling 40-120kW
      5-Layer Stack: Slurm vs K8s
      4 Pillars: Compute, Net, Storage, Support
    CPU vs GPU
      CPU: Low Latency, Sequential
      GPU: High Throughput, Parallel SIMT
      CUDA Cores Scalar vs Tensor Cores Matrix
      Precisions: FP32, TF32, FP16, BF16, FP8, INT8
      Pageable vs Pinned Memory & Pooling
    Network & 3D Parallelism
      Intra-node: NVLink & NVSwitch
      Inter-node: InfiniBand + GPUDirect RDMA
      Data vs Tensor vs Pipeline Parallelism + ZeRO
    Storage & Cloud
      NVMe + GPUDirect Storage GDS
      Parallel FS Lustre/WEKA vs NFS vs Object S3
      Cloud OpEx vs On-Prem CapEx vs Hybrid`,
                branches: [
                    { title: "⚙️ CPU vs. GPU & Chuẩn Số Học", items: ["<strong>CPU (Chuyên cơ):</strong> Low Latency, tuần tự | <strong>GPU (Airbus A380):</strong> High Throughput, song song", "<strong>Tensor Cores:</strong> Nhân-cộng ma trận $D = A \\times B + C$ tốc độ cao", "<strong>BF16:</strong> 8-bit mũ giống FP32 -> Dải động rộng hơn FP16, giảm mạnh nguy cơ tràn số đạo hàm", "<strong>FP8:</strong> Transformer Engine trên Hopper (H100) / Blackwell (B200)"] },
                    { title: "🌐 Mạng & 3D Parallelism", items: ["<strong>NVLink/NVSwitch:</strong> Nối GPU trong <strong>cùng 1 máy (Intra-node)</strong> -> Phù hợp cho <strong>Tensor Parallelism</strong>", "<strong>InfiniBand + GPUDirect RDMA:</strong> Nối GPU <strong>giữa các máy (Inter-node)</strong> giảm trung gian CPU", "<strong>Data Parallelism:</strong> Chia batch | <strong>Tensor:</strong> Cắt 1 tầng | <strong>Pipeline:</strong> Chia các tầng nối tiếp"] },
                    { title: "💾 Lưu Trữ & Triển Khai", items: ["<strong>GPUDirect Storage (GDS):</strong> DMA thẳng từ NVMe/Parallel FS vào GPU VRAM, bỏ qua CPU Bounce Buffer", "<strong>Parallel FS (Lustre/WEKA):</strong> Nhiều node đọc song song | <strong>NFS:</strong> Lưu code/config | <strong>S3:</strong> Cold data", "<strong>Cloud (OpEx):</strong> Linh hoạt | <strong>On-Prem (CapEx):</strong> Kiểm soát hạ tầng trực tiếp (cần cấu hình bảo mật chuẩn)"] }
                ]
            },
            mod02: {
                root: "🧠 Mindmap Module 02: AI/ML Fundamentals (Nền Tảng Học Máy)",
                mermaid: `mindmap
  root((Module 02: AI/ML))
    Phân cấp AI
      AI > ML > DL & GenAI
      DL: Automatic Feature Extraction
      Algorithm vs Model vs Inference
    Supervised Có nhãn
      Regression: Linear, MSE
      Classification: Logistic, SVM, Trees
      Bagging Random Forest: Giảm Variance
      Boosting XGBoost: Giảm Bias
    Unsupervised Không nhãn
      Clustering: K-Means, DBSCAN
      Dim Reduction: PCA, t-SNE, UMAP
    Reinforcement Learning
      Agent, Environment, Reward
    Bias-Variance & Reg
      Overfitting vs Underfitting
      L1 Lasso Feature Selection
      L2 Ridge Weight Decay`,
                branches: [
                    { title: "🎯 Supervised Learning (Có nhãn)", items: ["<strong>Regression (Dự đoán số thực):</strong> Linear Regression + MSE Loss", "<strong>Classification (Phân loại):</strong> Logistic Regression, SVM", "<strong>Random Forest (Bagging):</strong> Cây song song độc lập -> <strong>Giảm Variance (Giảm Overfitting)</strong>", "<strong>XGBoost (Boosting):</strong> Cây nối tiếp sửa lỗi -> <strong>Giảm Bias</strong>"] },
                    { title: "🔍 Unsupervised Learning (Không nhãn)", items: ["<strong>PCA (Principal Component Analysis):</strong> Giảm chiều dữ liệu (nén nhiều cột xuống các thành phần chính giữ nhiều phương sai nhất)", "<strong>K-Means:</strong> Phân cụm theo tâm K cụm", "<strong>DBSCAN:</strong> Phân cụm theo mật độ, tự động phát hiện điểm nhiễu (Outliers)"] },
                    { title: "⚖️ Bias-Variance & Khái Niệm Cốt Lõi", items: ["<strong>Overfitting (High Variance):</strong> Train tốt, Val kém -> Xử lý bằng <strong>L2 (Weight Decay), Dropout, Thêm Data</strong>", "<strong>L1 (Lasso):</strong> Ép trọng số kém quan trọng về <strong>$0$</strong> (Chọn lọc đặc trưng)", "<strong>Algorithm</strong> (Công thức) -> Huấn luyện ra <strong>Model</strong> (Trọng số) -> Chạy <strong>Inference</strong> (Dự đoán)"] }
                ]
            },
            mod03: {
                root: "✨ Mindmap Module 03: Generative AI & Foundation Models",
                mermaid: `mindmap
  root((Module 03: GenAI & LLM))
    Discriminative vs Generative
      Discriminative: P(Y|X) Ranh giới
      Generative: P(X) Tạo mới
      4 Họ: GANs, VAEs, Diffusion, Autoregressive
    Foundation Models
      Self-Supervised Learning
      Chinchilla Scaling Laws
      Emergent Abilities & Multimodal
    3 Giai Đoạn Huấn Luyện LLM
      1. Pre-training: Next-token, Base Model
      2. SFT: Instruction-following
      3. Alignment: RLHF PPO vs DPO vs RLAIF
    Tiến Hóa NLP
      One-Hot / TF-IDF
      Word2Vec: CBOW vs Skip-gram
      RNN / LSTM: Tuần tự theo bước thời gian
      Transformer: Song song vị trí token khi Train`,
                branches: [
                    { title: "🆚 Discriminative vs. Generative", items: ["<strong>Discriminative $P(Y|X)$:</strong> Phân loại, vẽ ranh giới (BERT, ResNet)", "<strong>Generative $P(X)$:</strong> Sinh dữ liệu mới (GANs: Generator đấu Discriminator; Diffusion: Khử nhiễu; Autoregressive: Đoán token kế tiếp)"] },
                    { title: "🚀 3 Giai Đoạn Huấn Luyện LLM", items: ["<strong>1. Pre-training:</strong> Học tự giám sát trên kho văn bản lớn -> Ra <strong>Base Model</strong>", "<strong>2. SFT (Supervised Fine-Tuning):</strong> Dạy làm theo lệnh trên cặp (Prompt, Response) -> Ra <strong>Instruct Model</strong>", "<strong>3. Alignment:</strong> <strong>RLHF (cần Reward Model + PPO)</strong> vs. <strong>DPO (Tối ưu trực tiếp Chosen/Rejected không cần Reward Model)</strong>"] },
                    { title: "📈 Tiến Hóa NLP & Word2Vec", items: ["<strong>Word2Vec CBOW:</strong> Nhìn các từ xung quanh đoán từ ở giữa (nhanh, từ phổ biến)", "<strong>Word2Vec Skip-gram:</strong> Nhìn 1 từ ở giữa đoán các từ xung quanh (tốt cho từ hiếm)", "<strong>Transformer vs RNN:</strong> Tính toán <strong>song song qua các vị trí token khi huấn luyện</strong> (còn pha Decode tự hồi quy vẫn sinh tuần tự từng token)"] }
                ]
            },
            mod04: {
                root: "⚡ Mindmap Module 04: Transformer Architecture (Kiến Trúc Transformer)",
                mermaid: `mindmap
  root((Module 04: Transformer))
    Tiền xử lý & Vị trí
      Subword Tokenization: BPE, WordPiece
      Positional Encoding: Sinusoidal, RoPE
    Cơ chế Attention
      Query Q, Key K, Value V
      Scaled Dot-Product Attention
      Bidirectional vs Masked Causal vs Cross
      MHA -> GQA + KV Cache + FlashAttention
    3 Biến thể & MoE
      Encoder-only: BERT Phân loại, NER, Embedding
      Decoder-only: GPT, LLaMA, Nemotron Sinh chữ
      Encoder-Decoder: T5, BART Dịch máy, Tóm tắt
      MoE: Router chọn Top-K Experts
    Sampling Controls
      Temperature, Top-K, Top-P`,
                branches: [
                    { title: "🔑 Giải Phẫu Attention ($Q, K, V$)", items: ["$\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$", "<strong>KV Cache:</strong> Lưu lại Key & Value của các token cũ để không phải tính lại khi sinh từ mới", "<strong>GQA (Grouped-Query Attention):</strong> Gom nhóm Key-Value giúp giảm mạnh VRAM của KV Cache", "<strong>RoPE (Rotary Positional Embedding):</strong> Mã hóa vị trí tương đối phổ biến trên LLaMA/Mistral"] },
                    { title: "🏛️ 3 Biến Thể Transformer & MoE", items: ["<strong>Encoder-only (BERT):</strong> Đọc 2 chiều -> Phân loại, Sentiment, <strong>NER</strong>, tạo Vector Embedding", "<strong>Decoder-only (GPT, LLaMA, Nemotron):</strong> Masked Causal 1 chiều -> Chatbot, Viết code, Suy luận", "<strong>Encoder-Decoder (T5, BART):</strong> <strong>Dịch máy (Translation)</strong>, <strong>Tóm tắt (Summarization)</strong>", "<strong>MoE (Mixtral):</strong> Mạng <strong>Router</strong> chỉ kích hoạt <strong>Top-K Experts</strong> trên mỗi token -> Giảm FLOPs mỗi token"] },
                    { title: "🎛️ Tham Số Điều Khiển Sinh Văn Bản", items: ["<strong>Temperature $\\approx 0$ (Greedy):</strong> Giảm tính ngẫu nhiên, ưu tiên token xác suất cao nhất (vẫn cần kiểm tra schema/unit test cho Code/SQL/JSON)", "<strong>Temperature cao ($>0.8$):</strong> Phẳng hóa xác suất, tăng độ đa dạng sáng tạo", "<strong>Top-K:</strong> Giữ cố định $K$ từ cao nhất | <strong>Top-P (Nucleus):</strong> Giữ tập từ động có tổng xác suất tích lũy $\\ge P$"] }
                ]
            },
            mod05: {
                root: "🎯 Mindmap Module 05: Model Selection & Evaluation Metrics",
                mermaid: `mindmap
  root((Module 05: Selection & Metrics))
    Chọn Model & Tốc độ
      SLM vs LLM & Open-Weights vs API
      Model Card & Benchmarks: MMLU, HumanEval, GSM8K
      Latency: TTFT Prefill vs TPOT Decode
    Classification Metrics
      Confusion Matrix: TP, FP, FN, TN
      Precision: Sợ bắt nhầm Spam/FP
      Recall: Sợ bỏ lọt Ung thư/Gian lận/FN
      F1-Score & PR-AUC: Dữ liệu lệch Imbalanced
    Regression Metrics
      MAE, MSE, RMSE, R-squared
    NLP & GenAI Metrics
      BLEU: Dịch máy Precision
      ROUGE: Tóm tắt Recall
      Cosine Similarity: Tương đồng Embedding
      RAG Eval: Recall@k, MRR, nDCG + Faithfulness`,
                branches: [
                    { title: "📊 Độ Đo Phân Loại & Hồi Quy", items: ["<strong>Không dùng Accuracy đơn lẻ</strong> cho dữ liệu mất cân bằng (Imbalanced Data) -> Xét <strong>Recall, Precision, F1-Score, PR-AUC</strong> theo chi phí sai lầm!", "<strong>Precision ($\\frac{TP}{TP+FP}$):</strong> Ưu tiên khi sợ bắt nhầm (Lọc thư rác Spam)", "<strong>Recall ($\\frac{TP}{TP+FN}$):</strong> Ưu tiên khi sợ bỏ lọt ca bệnh/gian lận (Y tế, An ninh)", "<strong>MSE / RMSE:</strong> Bình phương sai số nên phạt nặng các điểm <strong>Outliers</strong>"] },
                    { title: "🗣️ Độ Đo Ngôn Ngữ (NLP & RAG)", items: ["<strong>BLEU:</strong> Đánh giá <strong>Dịch máy (Machine Translation)</strong> (Thiên về Precision n-grams)", "<strong>ROUGE:</strong> Đánh giá <strong>Tóm tắt văn bản (Text Summarization)</strong> (Thiên về Recall bao phủ ý)", "<strong>Cosine Similarity:</strong> Đo độ gần giữa 2 Vector Embedding (bước truy xuất sơ bộ)", "<strong>Đánh giá RAG toàn diện:</strong> Retrieval (<strong>Recall@k, MRR, nDCG</strong>) + Generation (<strong>Faithfulness & Answer Relevance</strong>)"] },
                    { title: "⏱️ Tốc Độ Suy Luận & Kiểm Thử", items: ["<strong>TTFT (Time To First Token):</strong> Thời gian chờ ra chữ đầu tiên (Pha Prefill - Compute bound)", "<strong>TPOT / ITL:</strong> Thời gian sinh giữa 2 chữ liên tiếp (Pha Decode - Memory bandwidth bound)", "<strong>k-Fold Cross-Validation:</strong> Chia $k$ phần xoay vòng | <strong>Canary Rollout:</strong> Mở thử cho 1–5% user thật"] }
                ]
            },
            mod06: {
                root: "🛠️ Mindmap Module 06: Model Customization (Prompt, RAG, PEFT, Python API)",
                mermaid: `mindmap
  root((Module 06: Customization))
    Cấp 1: Prompt Engineering
      Zero-shot, One-shot, Few-shot
      Chain-of-Thought CoT & Self-Consistency
      ReAct: Reasoning + Acting gọi Tool
      System Prompt
    Cấp 2: RAG Kiến thức động
      Chunking + Overlap giữ ngữ cảnh
      Embedding + Vector DB HNSW/IVF
      Hybrid Search: BM25 + Vector + Reranker
      RAG Eval, Document ACLs & Anti-Injection
    Cấp 3: PEFT Fine-Tuning
      Đóng băng W0, giảm Catastrophic Forgetting
      LoRA: Ma trận hạng thấp A x B
      QLoRA: Nén gốc 4-bit NF4 + LoRA
    Python Dev Integration
      OpenAI/NIM Client + Timeout + TTFT Logging
      Pydantic JSON Schema Validation`,
                branches: [
                    { title: "💬 Cấp 1: Prompt Engineering & Python Dev", items: ["<strong>Zero / One / Few-shot, CoT, ReAct:</strong> Điều hướng mô hình không cần cập nhật trọng số", "<strong>Python Integration:</strong> Gọi endpoint OpenAI/NIM kèm `timeout`, đo <strong>TTFT & Latency</strong>", "<strong>Structured Output:</strong> Xác thực JSON trả về từ LLM bằng <strong>Pydantic (`BaseModel`)</strong>"] },
                    { title: "📚 Cấp 2: RAG, Đánh Giá & Bảo Mật Tài Liệu", items: ["<strong>Khi nào chọn RAG?</strong> Khi dữ liệu nội bộ <strong>cập nhật liên tục</strong>, cần <strong>trích dẫn nguồn</strong> và chi phí thấp", "<strong>Hybrid Search + Reranker:</strong> Kết hợp <strong>BM25 + Dense Vector</strong>, sau đó dùng <strong>Cross-Encoder Reranker</strong>", "<strong>Bảo mật RAG:</strong> Lọc phân quyền <strong>Document-level ACLs</strong> tại Vector DB và chắn <strong>Indirect Prompt Injection</strong>"] },
                    { title: "🧬 Cấp 3: PEFT (LoRA/QLoRA) & Distillation", items: ["<strong>LoRA:</strong> Đóng băng $W_0$, chỉ train $\\Delta W = B \\times A$ tại các `target_modules` (thường từ một phần nhỏ của $1\\%$ đến vài $\\%$ tham số tùy bậc `r`)", "<strong>QLoRA:</strong> Lượng tử hóa mô hình gốc xuống <strong>4-bit (`NF4`)</strong> rồi gắn Adapter LoRA", "<strong>Knowledge Distillation:</strong> Mô hình <strong>Teacher (lớn)</strong> truyền dạy tri thức cho <strong>Student (nhỏ)</strong>"] }
                ]
            },
            mod07: {
                root: "🔄 Mindmap Module 07: Model Training & Deployment Lifecycle",
                mermaid: `mindmap
  root((Module 07: Lifecycle))
    1-2. Data & EDA
      Imputation, Outliers, SMOTE trong Train Fold
      Min-Max Normalization vs Z-Score
      Text: Regex, Stemming vs Lemma, N-grams, Dedupe
    3. Model Training
      Learning Rate + Warmup/Decay
      Batch Size & Gradient Accumulation
      Epochs & Early Stopping chống Overfit
    4. Evaluation
      Quantitative vs Qualitative
      Perplexity PPL = exp Loss Càng thấp càng tốt
    5. Deploy & Optimization
      ONNX: Interoperability + Kiểm thử Parity
      Mixed Precision FP16/BF16 vs Quantization INT8/INT4
      Pruning: Cắt tỉa trọng số 2:4 Sparsity
      Data Drift P(X) vs Concept Drift P(Y|X)`,
                branches: [
                    { title: "🧹 Tiền Xử Lý Dữ Liệu & Chống Leakage", items: ["<strong>SMOTE & Scaling:</strong> Chỉ `.fit()` / tạo mẫu nhân tạo trên <strong>Training Fold</strong> sau khi tách dữ liệu để chống <strong>Data Leakage</strong>", "<strong>Regex:</strong> Xóa HTML tags, Email, SĐT cố định nhanh chóng | <strong>Deduplication:</strong> Loại bỏ văn bản trùng lặp", "<strong>Stemming</strong> (Cắt đuôi: <em>studies -> studi</em>) vs. <strong>Lemmatization</strong> (Từ gốc chuẩn: <em>better -> good</em>)"] },
                    { title: "🎛️ Hyperparameters & Perplexity", items: ["<strong>Learning Rate quá cao:</strong> Loss nhảy loạn / `NaN` | <strong>Quá thấp:</strong> Hội tụ chậm", "<strong>Early Stopping:</strong> Tự động dừng train khi <strong>Validation Loss bắt đầu tăng</strong> để chặn Overfitting", "<strong>Perplexity (PPL):</strong> Đo độ bối rối nội tại của LLM trên cùng bộ Tokenizer -> <strong>PPL càng thấp càng tốt!</strong>"] },
                    { title: "🚀 Tối Ưu Hóa Triển Khai & Giám Sát Drift", items: ["<strong>ONNX:</strong> Định dạng trung gian trao đổi mô hình (cần kiểm tra `opset`, `dynamic_axes` và <strong>numerical parity</strong>)", "<strong>Quantization:</strong> Chuyển xuống `FP8`/`INT8`/`INT4` (khác với Mixed Precision `FP16`/`BF16`) | <strong>Pruning:</strong> Cắt tỉa trọng số", "<strong>Data Drift:</strong> Phân phối đầu vào $P(X)$ đổi | <strong>Concept Drift:</strong> Quy luật thực tế $P(Y|X)$ đổi theo thời gian"] }
                ]
            },
            mod08: {
                root: "🟢 Mindmap Module 08: NVIDIA AI Ecosystem (Hệ Sinh Thái Toàn Diện NVIDIA)",
                mermaid: `mindmap
  root((Module 08: NVIDIA Stack))
    1. Phần Cứng & Quản Trị
      DGX: Volta -> Ampere -> Hopper -> Blackwell
      MIG: Tối đa 7 thực thể trên GPU hỗ trợ
      Grace CPU: 72 nhân Arm + NVLink-C2C
      Base Command & CUDA + nvidia-smi
    2. RAPIDS Data Science GPU
      cuDF Pandas | cuPy NumPy | cuIO
      cuML Sklearn | cuGraph NetworkX | cuVS
    3. NeMo Framework LLM
      Curator, Megatron, Customizer, Evaluator, Retriever, Guardrails
      Models: Nemotron Text, BioNeMo, Picasso
    4. Tối Ưu & Phục Vụ Suy Luận
      TensorRT & TensorRT-LLM: Layer Fusion, PagedAttention, In-flight Batching
      Triton Inference Server: Đa framework, Dynamic Batching
      NVIDIA NIM: Microservice + OpenAI-compatible API
      NGC Catalog & GPU Operator K8s`,
                branches: [
                    { title: "⚡ NVIDIA RAPIDS (Ánh Xạ CPU -> GPU)", items: ["`Pandas` $\\to$ <strong>`cuDF`</strong> (DataFrame trên GPU)", "`NumPy` $\\to$ <strong>`cuPy`</strong> | `PyArrow` $\\to$ <strong>`cuIO`</strong>", "`Scikit-learn` $\\to$ <strong>`cuML`</strong> (Thuật toán ML trên GPU)", "`NetworkX` $\\to$ <strong>`cuGraph`</strong> (Đồ thị) | `FAISS` $\\to$ <strong>`cuVS`</strong> (Vector Search)"] },
                    { title: "🛠️ NeMo Framework & TensorRT-LLM", items: ["<strong>NVIDIA NeMo:</strong> Framework xây dựng LLM (<strong>Curator</strong> lọc data, <strong>Megatron</strong> train phân tán, <strong>Customizer</strong> LoRA/DPO, <strong>Retriever</strong> RAG, <strong>Evaluator</strong>)", "<strong>TensorRT / TensorRT-LLM:</strong> SDK tối ưu hóa <strong>Suy luận</strong> với <strong>Layer Fusion, In-flight Batching, PagedAttention, FP8/INT8/INT4</strong>"] },
                    { title: "🌐 Triton Server vs. NIM vs. MIG", items: ["<strong>Triton Inference Server:</strong> Phần mềm máy chủ phục vụ mô hình đa framework (PyTorch, TensorRT, ONNX) trên cả GPU & CPU qua HTTP/gRPC", "<strong>NVIDIA NIM:</strong> Container Microservice đóng gói sẵn Model + Engine tối ưu + <strong>REST API tương thích chuẩn OpenAI cho các endpoint hỗ trợ</strong>", "<strong>NVIDIA MIG:</strong> Chia 1 GPU được hỗ trợ (như A100/H100/B200) thành tối đa <strong>7 thực thể GPU cách ly phần cứng</strong>"] }
                ]
            },
            mod09: {
                root: "🛡️ Mindmap Module 09: Ethical & Trustworthy AI (AI Đạo Đức & Đáng Tin Cậy)",
                mermaid: `mindmap
  root((Module 09: Trustworthy AI))
    Rủi Ro & Tấn Công
      Hallucination: Intrinsic vs Extrinsic
      Bias: Historical, Sampling, Measurement
      Direct vs Indirect Prompt Injection
      Jailbreaking & Data Poisoning
    4 Trụ Cột & Giải Pháp NVIDIA
      Privacy -> NVIDIA Confidential Computing TEE
      Safety -> NeMo Guardrails Colang + Embedding Classifier
      Transparency -> Explainable AI SHAP/LIME + Data Lineage
      Non-discrimination -> Diverse Data + Augmentation
    Kiểm Thử & Pháp Lý
      Red Teaming: NVIDIA Garak quét lỗ hổng
      NIST AI RMF & EU AI Act`,
                branches: [
                    { title: "⚠️ Rủi Ro & Các Cuộc Tấn Công LLM", items: ["<strong>Intrinsic Hallucination:</strong> Bịa mâu thuẫn với tài liệu nguồn | <strong>Extrinsic:</strong> Bịa thêm ngoài tài liệu", "<strong>Direct Prompt Injection:</strong> User gõ lệnh lừa trực tiếp vào ô chat", "<strong>Indirect Prompt Injection:</strong> Lệnh độc hại cài ẩn trong <strong>email, trang web, file PDF</strong> mà AI đọc phải"] },
                    { title: "🔒 Confidential Computing vs. NeMo Guardrails", items: ["<strong>Privacy (Bảo vệ dữ liệu đang tính toán - Data in Use):</strong> Dùng phần cứng <strong>NVIDIA Confidential Computing (Hopper/Blackwell TEE)</strong> kết hợp attestation & quản lý khóa", "<strong>Safety (Kiểm soát luồng hội thoại & nội dung):</strong> Dùng <strong>NeMo Guardrails</strong> (5 loại Rails: Input, Dialog, Retrieval, Execution, Output với <strong>Colang</strong>)", "<strong>Embedding Classifier:</strong> Phân loại câu hỏi đầu vào bằng Vector nhanh & nhẹ trước khi gọi LLM chính"] },
                    { title: "🔍 Minh Bạch (XAI), Công Bằng & Red Teaming", items: ["<strong>Transparency:</strong> Dùng <strong>Explainable AI (SHAP, LIME)</strong> giải thích đóng góp của từng biến đầu vào + <strong>Data Lineage</strong> truy vết nguồn gốc", "<strong>Non-discrimination:</strong> Dùng <strong>Diverse Datasets, Disaggregated Evaluation & Data Augmentation</strong>", "<strong>Red Teaming:</strong> Dùng công cụ <strong>NVIDIA Garak</strong> tự động quét lỗ hổng bảo mật LLM"] }
                ]
            },
            mod10: {
                root: "🔬 Mindmap Module 10: Deep Learning Mechanics, Diffusion & Additional Topics",
                mermaid: `mindmap
  root((Module 10: DL & Diffusion))
    Giải Phẫu Mạng Nơ-ron
      Forward -> Loss -> Backprop -> Optimizer AdamW
      Vanishing vs Exploding Gradients
      Skip Connections ResNet & Gradient Clipping
    Hàm Kích Hoạt & Chuẩn Hóa
      Hidden Layers: ReLU, GELU, SwiGLU
      Output Layer: Sigmoid Nhị phân vs Softmax Đa lớp
      LayerNorm/RMSNorm Transformer vs BatchNorm CNN
      Dropout chống Overfitting
    Diffusion & Vision
      Forward thêm nhiễu vs Reverse khử nhiễu
      Latent Diffusion: VAE + U-Net/DiT + CLIP + CFG Scale
      Vision Transformer ViT: Chia ảnh thành Patches
    NLP Tasks, RAD & Hugging Face
      NER: PER, ORG, LOC
      Extractive vs Abstractive Summarization
      RAD Prototyping & Hugging Face Hub`,
                branches: [
                    { title: "⚡ Đạo Hàm & Hàm Kích Hoạt (Activations)", items: ["<strong>Vanishing Gradients (Triệt tiêu đạo hàm):</strong> Xử lý bằng <strong>ReLU / GELU / SwiGLU</strong> + <strong>Skip Connections ($y = F(x) + x$)</strong>", "<strong>Exploding Gradients (Bùng nổ đạo hàm / Loss NaN):</strong> Xử lý bằng <strong>Gradient Clipping</strong> + Giảm Learning Rate", "<strong>Sigmoid:</strong> Đầu ra nhị phân $(0,1)$ | <strong>Softmax:</strong> Đầu ra đa lớp & từ vựng LLM (Tổng xác suất $= 1.0$)"] },
                    { title: "🎨 Latent Diffusion Models (Stable Diffusion) & ViT", items: ["<strong>2 Pha Diffusion:</strong> <strong>Forward</strong> (Cộng nhiễu Gaussian từ từ) $\\leftrightarrow$ <strong>Reverse</strong> (Mạng <strong>U-Net/DiT</strong> dự đoán và trừ nhiễu để tạo ảnh)", "<strong>Latent Diffusion:</strong> <strong>CLIP</strong> (Mã hóa text) + <strong>VAE</strong> (Nén ảnh vào không gian tiềm ẩn Latent) + <strong>U-Net/DiT</strong> (Khử nhiễu) + <strong>CFG Scale</strong> (Độ bám sát prompt)", "<strong>Vision Transformer (ViT):</strong> Cắt ảnh thành các ô vuông <strong>Image Patches ($16\\times 16$)</strong> coi như chuỗi token"] },
                    { title: "🤗 NER, RAD & Hệ Sinh Thái Hugging Face", items: ["<strong>NER (Named Entity Recognition):</strong> Trích xuất Tên người (`PER`), Tổ chức (`ORG`), Địa điểm (`LOC`)", "<strong>RAD (Rapid Application Development):</strong> Làm mẫu thử (Prototype) nhanh lấy phản hồi sớm", "<strong>Hugging Face:</strong> `transformers` (Models), `datasets` (Parquet/Arrow), `peft` (LoRA), `accelerate`"] }
                ]
            },
            mod11: {
                root: "📐 Mindmap Module 11: Thiết Kế Thực Nghiệm, Trực Quan Hóa & Chống Data Leakage",
                mermaid: `mindmap
  root((Module 11: Experiment & Data Vis))
    1. Thiết Kế Thực Nghiệm
      Giả thuyết H0 vs H1
      Biến Độc lập, Phụ thuộc & Ngoại lai Confounding
      Control Baseline vs Treatment A/B Test
      Ablation Study: Gỡ từng thành phần
      Statistical p-value vs Practical Significance
      PyTorch Reproducibility & Random Seeds
    2. Phân Tích & Trực Quan Hóa
      Skewed Data: Median & IQR bền vững hơn Mean
      Outliers: Quy tắc 1.5 x IQR trên Box Plot
      Missing Data: MCAR, MAR, MNAR + Cờ báo khuyết
      Charts: Histogram, Box Plot, Scatter, Heatmap
      PR Curve PR-AUC vs ROC Curve trên dữ liệu hiếm
      Correlation vs Causation & Simpson Paradox
    3. Chống Data Leakage
      Chia Train/Val/Test TRƯỚC khi fit Scaler/Imputer
      sklearn Pipeline trong Cross-Validation
      SMOTE chỉ chạy trong Training Fold
      TimeSeriesSplit & GroupKFold`,
                branches: [
                    { title: "🧪 Thiết Kế Thực Nghiệm Khoa Học (22%)", items: ["<strong>Kiểm soát biến ngoại lai (Confounding Variables):</strong> Chỉ thay đổi 1 biến độc lập khi so sánh với <strong>Baseline / Control</strong>", "<strong>Ablation Study:</strong> Tắt từng thành phần (VD: Reranker, Hybrid Search) để đo đóng góp thực tế về chất lượng & độ trễ", "<strong>Statistical vs. Practical Significance:</strong> $p < 0.05$ chưa đủ nếu mức tăng chất lượng quá nhỏ so với chi phí/độ trễ tăng vọt"] },
                    { title: "📊 Thống Kê & Chọn Biểu Đồ (14%)", items: ["<strong>Dữ liệu lệch (Skewed) / có Outliers:</strong> Dùng <strong>Median (Trung vị)</strong> và <strong>IQR ($Q_3 - Q_1$)</strong> thay vì Mean", "<strong>Chọn biểu đồ:</strong> Phân phối 1 biến $\\to$ <strong>Histogram</strong> | So sánh nhóm & Outliers $\\to$ <strong>Box Plot</strong> | Tương quan 2 biến $\\to$ <strong>Scatter Plot</strong> | Ma trận $\\to$ <strong>Heatmap</strong>", "<strong>Dữ liệu cực lệch:</strong> Ưu tiên <strong>PR Curve (`PR-AUC`)</strong> hơn `ROC-AUC`; không nhầm <strong>Correlation</strong> thành <strong>Causation</strong>"] },
                    { title: "🛡️ Quy Tắc Chống Data Leakage", items: ["<strong>Quy tắc cốt lõi:</strong> Luôn chia `Train / Val / Test` <strong>TRƯỚC</strong>, sau đó chỉ `.fit()` bộ `Scaler / Imputer / Tokenizer` trên <strong>Train</strong>", "<strong>Cross-Validation & SMOTE:</strong> Đặt `StandardScaler` và `SMOTE` bên trong `Pipeline` để chỉ chạy trên <strong>Training Fold</strong>", "<strong>Dữ liệu đặc thù:</strong> Chuỗi thời gian dùng `TimeSeriesSplit`; nhiều mẫu cùng 1 bệnh nhân/user dùng `GroupKFold`"] }
                ]
            },
            exam: {
                root: "🏆 Chiến Thuật Phòng Thi, 10 Bảng Phản Xạ & Đề Luyện Tập (30 Câu + 50 Câu Định Giờ)",
                mermaid: `mindmap
  root((Chiến Thuật Phòng Thi))
    Quản lý 60 Phút
      Pass 1: 45s/câu bắt từ khóa
      Pass 2: Flag câu dài quay lại sau
      Không bỏ trống bất kỳ câu nào
    Mẹo Loại Trừ Đáp Án Nhiễu
      Triton là phần mềm serving, không dùng để train
      Imbalanced Data không dùng Accuracy đơn lẻ
      Tài liệu đổi liên tục ngân sách thấp -> Chọn RAG
    Từ Khóa Trọng Tâm
      Dịch máy -> T5 + BLEU
      Tóm tắt -> BART/T5 + ROUGE
      NER / Embedding -> BERT Encoder-only
      Độ bối rối LLM -> Perplexity càng thấp càng tốt`,
                branches: [
                    { title: "🚫 3 Bẫy Loại Trừ Nhanh", items: ["Thấy đáp án nói <em>'Triton là phần cứng GPU'</em> hoặc <em>'Triton dùng để train mô hình'</em> -> <strong>Loại ngay!</strong>", "Thấy bài toán dữ liệu lệch (gian lận 0.1%, bệnh hiếm 2%) mà chỉ nhìn <em>'Accuracy'</em> -> <strong>Loại ngay, xét Recall / Precision / F1 / PR-AUC!</strong>", "Thấy cần tra cứu tài liệu nội bộ cập nhật hàng ngày mà chọn <em>'Train from scratch / Full Fine-Tuning'</em> -> <strong>Ưu tiên chọn RAG!</strong>"] },
                    { title: "⚡ Phản Xạ Từ Khóa Hạ Tầng & RAPIDS", items: ["<strong>Intra-node GPU:</strong> NVLink + Tensor Parallelism | <strong>Inter-node GPU:</strong> InfiniBand + GPUDirect RDMA", "<strong>Storage -> VRAM:</strong> GPUDirect Storage (GDS) | <strong>Chia 1 GPU hỗ trợ thành tối đa 7:</strong> NVIDIA MIG", "<strong>DataFrame GPU:</strong> `cuDF` | <strong>ML GPU:</strong> `cuML` | <strong>Đồ thị GPU:</strong> `cuGraph`"] },
                    { title: "🛡️ Phản Xạ Thực Nghiệm & Bảo Mật", items: ["<strong>Chống Data Leakage:</strong> Chia Train/Test trước, bọc `Scaler` & `SMOTE` trong `Pipeline`", "<strong>Bảo vệ dữ liệu đang xử lý (Data in Use):</strong> NVIDIA Confidential Computing (TEE)", "<strong>Chặn prompt độc hại & bảo vệ RAG:</strong> NeMo Guardrails (5 loại Rails) + Document-level ACLs"] }
                ]
            }
        };

        function parseMockQuestions(mode = 'exam50_en') {
            if (mode === 'practice01') {
                return PRACTICE_01_DATA.map((item, idx) => ({
                    num: idx + 1,
                    title: `Question ${idx + 1} (${item.topic} · ${item.difficulty})`,
                    domain: item.domain,
                    prompt: item.question,
                    options: item.options.map(opt => ({ letter: opt.id, text: opt.text })),
                    answer: [...item.correctAnswers].sort().join(','),
                    multiple: item.type === 'multiple',
                    explanation: `<strong>Correct Answer: ${item.correctAnswers.join(', ')}</strong><br/>${item.explanation}<br/>${item.options.map(opt => `<strong>${opt.id}.</strong> ${opt.explanation}`).join('<br/>')}`
                }));
            }
            if ((mode === 'exam50_en' || mode === 'exam50_vi' || mode === 'set50') && EXAM_50_BILINGUAL.length > 0) {
                const isEn = (mode === 'exam50_en');
                return EXAM_50_BILINGUAL.map(item => {
                    const optsObj = isEn ? item.options_en : item.options_vi;
                    const options = ['A', 'B', 'C', 'D'].map(letter => ({
                        letter,
                        text: optsObj[letter] || ''
                    }));
                    const explanation = isEn
                        ? `<strong>Correct Answer: ${item.ans}</strong><br/>• ${item.exp_en}`
                        : `<strong>Đáp án đúng: ${item.ans}</strong><br/>• <strong>Tại sao ${item.ans} đúng:</strong> ${item.why_correct_vi}<br/>• <strong>Tại sao các phương án nhiễu sai:</strong> ${item.why_wrong_vi}`;
                    return {
                        num: item.num,
                        title: isEn ? `Question ${item.num} (${item.topic})` : `Bài ${item.num} (${item.topic})`,
                        domain: item.domain,
                        prompt: isEn ? item.q_en : item.q_vi,
                        options,
                        answer: item.ans,
                        explanation
                    };
                });
            }

            const examDoc = DOCS_DATA.find(d => d.id === 'exam');
            if (!examDoc) return [];
            const lines = examDoc.content.split(/\r?\n/);
            const questions30 = [];
            const questions50 = [];
            const explanations50 = {};

            // First pass: extract Section 5 explanations for Bài 1..50
            let currentExpNum = null;
            for (let i = 0; i < lines.length; i++) {
                const rawLine = lines[i];
                const trimmed = rawLine.trim();
                const headerMatch = trimmed.match(/^\*\s+\*\*Bài\s+(\d+)\s+—\s+Đáp án đúng:\s*([A-D])\*\*/);
                if (headerMatch) {
                    currentExpNum = headerMatch[1];
                    explanations50[currentExpNum] = {
                        answer: headerMatch[2],
                        explanation: `<strong>Đáp án đúng: ${headerMatch[2]}</strong>`
                    };
                } else if (currentExpNum && (rawLine.startsWith("  * ") || rawLine.startsWith("    * "))) {
                    explanations50[currentExpNum].explanation += "<br/>• " + trimmed.replace(/^\*\s+/, "");
                } else if (trimmed.startsWith("## 6.")) {
                    currentExpNum = null;
                }
            }

            // Second pass: extract 30-question and 50-question sets
            let currentDomain = "General";
            let currentQ = null;
            let currentTarget = null;

            for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();
                if (line.startsWith("<details>") || line.startsWith("</details>") || line.startsWith("<summary>")) {
                    continue;
                }
                if (line.startsWith("### PHẦN ") || line.startsWith("### MIỀN ")) {
                    currentDomain = line.replace("### ", "");
                } else if (line.startsWith("### Bảng Tra Cứu")) {
                    if (currentQ && currentTarget) currentTarget.push(currentQ);
                    currentQ = null;
                    currentTarget = null;
                } else if (line.startsWith("#### Câu ")) {
                    if (currentQ && currentTarget) currentTarget.push(currentQ);
                    currentTarget = questions30;
                    currentQ = {
                        title: line.replace("#### ", ""),
                        domain: currentDomain,
                        prompt: "",
                        options: [],
                        answer: "",
                        explanation: ""
                    };
                } else if (line.startsWith("#### Bài ")) {
                    if (currentQ && currentTarget) currentTarget.push(currentQ);
                    currentTarget = questions50;
                    const titleText = line.replace("#### ", "");
                    const numMatch = titleText.match(/^Bài\s+(\d+)/);
                    const bNum = numMatch ? numMatch[1] : null;
                    const expObj = (bNum && explanations50[bNum]) ? explanations50[bNum] : { answer: "", explanation: "" };
                    currentQ = {
                        title: titleText,
                        domain: currentDomain,
                        prompt: "",
                        options: [],
                        answer: expObj.answer,
                        explanation: expObj.explanation
                    };
                } else if (line.startsWith("## 5.")) {
                    if (currentQ && currentTarget) currentTarget.push(currentQ);
                    currentQ = null;
                    currentTarget = null;
                } else if (currentQ) {
                    const optMatch = line.match(/^\*\s+\*\*([A-D])\.\*\*\s+(.*)$/);
                    if (optMatch) {
                        currentQ.options.push({ letter: optMatch[1], text: optMatch[2] });
                    } else if (line.startsWith("> **Đáp án đúng:")) {
                        const ansMatch = line.match(/\*\*Đáp án đúng:\s*([A-D])/);
                        if (ansMatch) currentQ.answer = ansMatch[1];
                        currentQ.explanation = line.replace(/^>\s*/, "");
                    } else if (line.startsWith("> ") && currentQ.explanation && currentTarget === questions30) {
                        currentQ.explanation += "<br/>" + line.replace(/^>\s*/, "");
                    } else if (line !== "---" && !line.startsWith("### ") && !line.startsWith("## ") && currentQ.options.length === 0) {
                        if (line) currentQ.prompt += (currentQ.prompt ? " " : "") + line;
                    }
                }
            }
            if (currentQ && currentTarget) currentTarget.push(currentQ);

            if (mode === 'set30') return questions30;
            if (mode === 'all80') {
                const q50 = EXAM_50_BILINGUAL.length > 0 ? parseMockQuestions('set50') : questions50;
                return [...q50, ...questions30];
            }
            return questions50;
        }

        // State Management
        let currentView = 'reader';
        let currentDocId = 'readme';
        let completedDocs = JSON.parse(localStorage.getItem('nca_genl_completed') || '[]');
        let fontSize = 16;

        mermaid.initialize({
            startOnLoad: false,
            theme: 'neutral',
            securityLevel: 'loose',
            fontFamily: 'Inter, sans-serif'
        });

        // Markdown Preprocessing with CustomDiagramEngine
        function renderMarkdownToHtml(mdText) {
            let processed = mdText.replace(/(?:file:\/\/\/[^)]*?\/|\.\.?\/)([A-Za-z0-9_]+\.md)(#[A-Za-z0-9_-]+)?/g, (match, filename) => {
                const found = DOCS_DATA.find(d => d.file.toLowerCase() === filename.toLowerCase());
                return found ? `#jump-${found.id}` : match;
            });

            // Protect math blocks
            const mathBlocks = [];
            processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (m) => {
                mathBlocks.push(m);
                return `@@MATH_BLOCK_${mathBlocks.length - 1}@@`;
            });
            processed = processed.replace(/\$([^\$\n]+?)\$/g, (m) => {
                mathBlocks.push(m);
                return `@@MATH_INLINE_${mathBlocks.length - 1}@@`;
            });

            // Extract mermaid blocks and render with CustomDiagramEngine!
            const customDiagramHtmlBlocks = [];
            processed = processed.replace(/```mermaid\r?\n([\s\S]*?)```/g, (m, code) => {
                const rendered = CustomDiagramEngine.renderBlock(code.trim());
                customDiagramHtmlBlocks.push(rendered);
                return `@@CUSTOM_DIAGRAM_BLOCK_${customDiagramHtmlBlocks.length - 1}@@`;
            });

            let html = marked.parse(processed);

            // Transform GitHub blockquotes
            const alertIcons = {
                IMPORTANT: '💡 QUAN TRỌNG (IMPORTANT)',
                TIP: '🔥 MẸO PHÒNG THI & DEV (TIP)',
                WARNING: '⚠️ CẢNH BÁO BẪY ĐỀ THI (WARNING)',
                CAUTION: '🚨 LƯU Ý ĐẶC BIỆT (CAUTION)',
                NOTE: '📌 GHI CHÚ (NOTE)'
            };

            html = html.replace(/<blockquote>\s*<p>\[!(IMPORTANT|TIP|WARNING|CAUTION|NOTE)\]\s*(?:<br>|\n)?([\s\S]*?)<\/blockquote>/gi, (m, type, body) => {
                const key = type.toUpperCase();
                return `<div class="gh-alert ${key.toLowerCase()}">
                    <div class="gh-alert-title">${alertIcons[key] || key}</div>
                    <div><p>${body}</div>
                </div>`;
            });

            html = html.replace(/<table>/g, '<div class="table-wrapper"><table>').replace(/<\/table>/g, '</table></div>');

            // Restore Custom Diagram HTML blocks
            html = html.replace(/<p>@@CUSTOM_DIAGRAM_BLOCK_(\d+)@@<\/p>|@@CUSTOM_DIAGRAM_BLOCK_(\d+)@@/g, (m, i1, i2) => {
                const idx = i1 !== undefined ? i1 : i2;
                return customDiagramHtmlBlocks[Number(idx)];
            });

            // Restore Math blocks
            html = html.replace(/@@MATH_(?:BLOCK|INLINE)_(\d+)@@/g, (m, idx) => mathBlocks[Number(idx)]);

            return html;
        }

        function renderSidebar() {
            const listEl = document.getElementById('moduleListContainer');
            listEl.innerHTML = DOCS_DATA.map(doc => {
                const isDone = completedDocs.includes(doc.id);
                const isActive = currentView === 'reader' && currentDocId === doc.id;
                return `
                    <button class="module-item ${isActive ? 'active' : ''}" onclick="selectDocument('${doc.id}')">
                        <span class="module-icon">${doc.icon}</span>
                        <div class="module-info">
                            <div class="module-badge-row">
                                <span>${doc.badge}</span>
                                <span>${doc.weight}</span>
                            </div>
                            <div class="module-title" title="${doc.shortTitle}">${doc.shortTitle}</div>
                        </div>
                        <div class="check-circle ${isDone ? 'done' : ''}" onclick="event.stopPropagation(); toggleComplete('${doc.id}')" title="Đánh dấu đã học xong">✓</div>
                    </button>
                `;
            }).join('');

            const doneCount = completedDocs.length;
            const total = DOCS_DATA.length;
            document.getElementById('progressText').textContent = `${doneCount} / ${total} hoàn thành (${Math.round(doneCount/total*100)}%)`;
            document.getElementById('progressBarFill').style.width = `${(doneCount / total) * 100}%`;
        }

        function toggleComplete(docId) {
            if (completedDocs.includes(docId)) {
                completedDocs = completedDocs.filter(id => id !== docId);
            } else {
                completedDocs.push(docId);
            }
            localStorage.setItem('nca_genl_completed', JSON.stringify(completedDocs));
            renderSidebar();
            if (currentView === 'reader' && currentDocId === docId) {
                const btn = document.getElementById('heroCompleteBtn');
                if (btn) {
                    const isDone = completedDocs.includes(docId);
                    btn.className = `complete-btn ${isDone ? 'completed' : ''}`;
                    btn.innerHTML = isDone ? '✓ Đã hoàn thành tài liệu này' : '○ Đánh dấu đã học xong';
                }
            }
        }

        function closeMobileDrawers() {
            if (window.innerWidth <= 860) {
                document.getElementById('sidebarLeft').classList.add('collapsed');
            }
            document.getElementById('sidebarRight').classList.remove('mobile-open');
            document.getElementById('sidebarBackdrop').classList.remove('active');
        }

        function syncMobileFabState() {
            const tocFab = document.getElementById('openTocFab');
            const closeTocBtn = document.getElementById('closeTocMobileBtn');
            if (tocFab) tocFab.style.display = currentView === 'reader' ? 'flex' : 'none';
            if (closeTocBtn) closeTocBtn.style.display = window.innerWidth <= 1100 ? 'flex' : 'none';
        }

        function selectDocument(docId, scrollTargetText = null) {
            currentDocId = docId;
            if (currentView !== 'reader') {
                switchView('reader', false);
            }
            closeMobileDrawers();
            renderSidebar();
            renderReaderView(scrollTargetText);
        }

        function renderMindmapCardHtml(docId) {
            const mm = MODULE_MINDMAPS[docId];
            if (!mm) return '';
            const customMindmapDiagram = CustomDiagramEngine.renderBlock(mm.mermaid, `Sơ đồ Nhánh Tư duy Trực quan (${docId.toUpperCase()})`);
            return `
                <div class="mindmap-card">
                    <div class="mindmap-card-header" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
                        <div class="mindmap-card-title">🧠 SƠ ĐỒ TƯ DUY & TÓM TẮT CỐT LÕI (Bấm để Ẩn/Hiện nhanh)</div>
                        <span style="font-size:0.78rem; color:var(--text-muted)">Tóm tắt trực quan toàn bộ trọng tâm thi của phần này ▼</span>
                    </div>
                    <div class="mindmap-card-body">
                        <div class="tree-mindmap">
                            <div><span class="mm-root">${mm.root}</span></div>
                            <div class="mm-branches">
                                ${mm.branches.map(b => `
                                    <div class="mm-branch">
                                        <div class="mm-branch-title">${b.title}</div>
                                        <ul class="mm-leaf-list">
                                            ${b.items.map(item => `<li class="mm-leaf">${item}</li>`).join('')}
                                        </ul>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        <div style="margin-top:16px;">
                            ${customMindmapDiagram}
                        </div>
                    </div>
                </div>
            `;
        }

        // View 1: Full Document Reader
        async function renderReaderView(scrollTargetText = null) {
            const doc = DOCS_DATA.find(d => d.id === currentDocId) || DOCS_DATA[0];
            const isDone = completedDocs.includes(doc.id);
            const container = document.getElementById('contentContainer');
            document.getElementById('sidebarRight').style.display = '';

            const currentIndex = DOCS_DATA.findIndex(d => d.id === doc.id);
            const prevDoc = currentIndex > 0 ? DOCS_DATA[currentIndex - 1] : null;
            const nextDoc = currentIndex < DOCS_DATA.length - 1 ? DOCS_DATA[currentIndex + 1] : null;

            container.innerHTML = `
                <div class="module-hero">
                    <div>
                        <div class="module-hero-meta">
                            <span class="brand-badge">${doc.badge}</span>
                            <span class="domain-pill">Miền kiến thức: ${doc.domain}</span>
                            <span style="font-size:0.8rem; color:var(--text-muted)">📄 File gốc: <code>${doc.file}</code></span>
                        </div>
                        <h2 style="font-size:1.35rem; font-weight:800; color:var(--text-primary);">${doc.icon} ${doc.shortTitle}</h2>
                    </div>
                    <button class="complete-btn ${isDone ? 'completed' : ''}" id="heroCompleteBtn" onclick="toggleComplete('${doc.id}')">
                        ${isDone ? '✓ Đã hoàn thành tài liệu này' : '○ Đánh dấu đã học xong'}
                    </button>
                </div>

                ${renderMindmapCardHtml(doc.id)}

                <article class="markdown-body" id="markdownArticle">
                    ${renderMarkdownToHtml(doc.content)}
                </article>

                <div class="doc-footer-nav">
                    ${prevDoc ? `<button class="complete-btn" onclick="selectDocument('${prevDoc.id}')">← Bài trước: ${prevDoc.badge} (${prevDoc.shortTitle})</button>` : '<div></div>'}
                    ${nextDoc ? `<button class="complete-btn" style="margin-left:auto;" onclick="selectDocument('${nextDoc.id}')">Bài tiếp theo: ${nextDoc.badge} (${nextDoc.shortTitle}) →</button>` : ''}
                </div>
            `;

            container.querySelectorAll('a[href^="#jump-"]').forEach(a => {
                a.addEventListener('click', (e) => {
                    e.preventDefault();
                    const targetId = a.getAttribute('href').replace('#jump-', '');
                    selectDocument(targetId);
                });
            });

            buildTOC();
            syncMobileFabState();
            await postProcessRenderedDOM(container);

            const mainScroll = document.getElementById('mainScroll');
            if (scrollTargetText) {
                setTimeout(() => {
                    const headings = container.querySelectorAll('h1, h2, h3, h4, p, td, li');
                    for (const el of headings) {
                        if (el.textContent.toLowerCase().includes(scrollTargetText.toLowerCase())) {
                            el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            el.style.transition = 'background 0.5s';
                            el.style.background = 'var(--nv-green-glow)';
                            setTimeout(() => { el.style.background = ''; }, 2200);
                            break;
                        }
                    }
                }, 150);
            } else {
                mainScroll.scrollTop = 0;
            }
        }

        function buildTOC() {
            const article = document.getElementById('markdownArticle');
            const tocList = document.getElementById('tocList');
            if (!article || !tocList) return;

            const headings = article.querySelectorAll('h2, h3');
            tocList.innerHTML = '';
            headings.forEach((h, index) => {
                const id = `sec-heading-${index}`;
                h.id = id;
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.href = `#${id}`;
                a.className = `toc-link ${h.tagName === 'H3' ? 'toc-h3' : ''}`;
                a.textContent = h.textContent;
                a.onclick = (e) => {
                    e.preventDefault();
                    h.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    tocList.querySelectorAll('.toc-link').forEach(l => l.classList.remove('active'));
                    a.classList.add('active');
                    if (window.innerWidth <= 1100) {
                        document.getElementById('sidebarRight').classList.remove('mobile-open');
                        document.getElementById('sidebarBackdrop').classList.remove('active');
                    }
                };
                li.appendChild(a);
                tocList.appendChild(li);
            });
        }

        async function postProcessRenderedDOM(container) {
            if (window.hljs) {
                container.querySelectorAll('pre code:not(.mermaid)').forEach(block => {
                    hljs.highlightElement(block);
                });
            }
            if (window.renderMathInElement) {
                renderMathInElement(container, {
                    delimiters: [
                        { left: '$$', right: '$$', display: true },
                        { left: '$', right: '$', display: false },
                        { left: '\\(', right: '\\)', display: false },
                        { left: '\\[', right: '\\]', display: true }
                    ],
                    throwOnError: false
                });
            }
        }

        // View 2: All Mindmaps Gallery Hub
        async function renderMindmapsHubView() {
            const container = document.getElementById('contentContainer');
            document.getElementById('sidebarRight').style.display = 'none';

            container.innerHTML = `
                <div class="module-hero">
                    <div>
                        <span class="brand-badge">BẢN ĐỒ TƯ DUY TOÀN DIỆN</span>
                        <h2 style="font-size:1.5rem; font-weight:800; margin-top:6px;">🧠 Hệ Thống Mindmap Tổng Hợp Toàn Bộ 11 Modules & 5 Miền Kiến Thức</h2>
                        <p style="color:var(--text-secondary); margin-top:4px;">Nhìn toàn cảnh kiến thức NCA-GENL trên một trang duy nhất với sơ đồ nhánh hiện đại. Bấm vào từng Module để nhảy tới bài học.</p>
                    </div>
                </div>

                ${DOCS_DATA.map(doc => `
                    <div style="margin-bottom: 32px;">
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; flex-wrap:wrap; gap:10px;">
                            <h3 style="color:var(--text-primary); font-size:1.15rem;">${doc.icon} ${doc.badge}: ${doc.shortTitle}</h3>
                            <button class="complete-btn" onclick="selectDocument('${doc.id}')">📖 Đọc chi tiết ${doc.badge} →</button>
                        </div>
                        ${renderMindmapCardHtml(doc.id)}
                    </div>
                `).join('')}
            `;

            await postProcessRenderedDOM(container);
            document.getElementById('mainScroll').scrollTop = 0;
        }

        // View 3: Decision Flowcharts & Interactive VRAM Calculator Hub
        async function renderDiagramsHubView() {
            const container = document.getElementById('contentContainer');
            document.getElementById('sidebarRight').style.display = 'none';

            const diag1 = CustomDiagramEngine.renderBlock(`flowchart TD
    Start(["Bài toán Doanh nghiệp cần tùy biến LLM"]) --> Q1{"1. Có cần cập nhật kiến thức tài liệu nội bộ liên tục & trích dẫn nguồn giảm ảo giác?"}
    Q1 -- "CÓ (Không đổi Weights)" --> RAG["✅ Chọn RAG (Retrieval-Augmented Generation)<br/>• Chunking + Overlap (10-20%)<br/>• Embedding + Vector DB (HNSW/IVF) + Document ACLs<br/>• Hybrid Search (BM25 + Vector) + Cross-Encoder Reranker"]
    Q1 -- "KHÔNG (Dạy kỹ năng / văn phong)" --> Q2{"2. Cần thử nghiệm nhanh chỉ bằng câu lệnh / vài ví dụ mẫu?"}
    Q2 -- "CÓ" --> Prompt["✅ Chọn Prompt Engineering<br/>• Zero-shot / Few-shot<br/>• Chain-of-Thought (CoT)<br/>• ReAct (Reasoning + Acting)"]
    Q2 -- "KHÔNG (Cần Fine-tune)" --> Q3{"3. Ràng buộc phần cứng VRAM & Mục tiêu triển khai?"}
    Q3 -- "Truyền tri thức sang SLM nhỏ chạy trên Edge/Mobile" --> Distill["✅ Chọn Knowledge Distillation<br/>• Mô hình Teacher (70B) truyền dạy cho Student (8B)"]
    Q3 -- "Có ít VRAM, đóng băng trọng số gốc" --> LoRA["✅ Chọn PEFT: LoRA / QLoRA<br/>• Đóng băng W0, chỉ train ma trận hạng thấp B x A (tỷ lệ tham số nhỏ tùy r & target_modules)<br/>• QLoRA: Nén thêm mô hình gốc xuống chuẩn 4-bit NF4"]`, "Cây Quyết Định 1: Chọn Phương Pháp Tùy Biến LLM (Module 3 & 6)");

            const diag2 = CustomDiagramEngine.renderBlock(`flowchart TD
    subgraph Cls["1. Bài toán Phân loại (Classification)"]
        C1["Dữ liệu cân bằng: Có thể dùng Accuracy"]
        C2["Dữ liệu mất cân bằng (Imbalanced: 98% vs 2%):<br/>• Không dùng Accuracy đơn lẻ -> Kết hợp F1-Score & PR-AUC (kèm ROC-AUC)<br/>• Sợ bắt nhầm (Lọc Spam / FP cao) -> Tối ưu Precision<br/>• Sợ bỏ lọt (Chẩn đoán Bệnh hiếm, Gian lận / FN cao) -> Tối ưu Recall"]
        C1 --> C2
    end
    subgraph Reg["2. Bài toán Hồi quy (Regression - Dự đoán số thực)"]
        R1["MSE & RMSE: Bình phương sai số -> Phạt nặng các điểm ngoại lai (Outliers)"]
        R2["MAE: Trị tuyệt đối sai số -> Bền vững với Outliers | R-squared: Tỷ lệ phương sai giải thích được"]
        R1 --> R2
    end
    subgraph NLP["3. Bài toán Ngôn ngữ, Dịch máy, Tóm tắt & RAG"]
        N1["Dịch máy (Translation) ➡️ Chọn BLEU (Thiên về Precision n-grams)"]
        N2["Tóm tắt văn bản (Summarization) ➡️ Chọn ROUGE (Thiên về Recall bao phủ ý)"]
        N3["Đánh giá RAG ➡️ Cosine Similarity (Vector) + Recall@k, MRR, nDCG (Retrieval) + Faithfulness & Answer Relevance"]
        N4["Độ bối rối nội tại của LLM ➡️ Chọn Perplexity (PPL = exp(Loss) — Càng THẤP càng tốt!)"]
        N1 --> N2 --> N3 --> N4
    end
    subgraph Lat["4. Tốc độ Suy luận LLM trên Production"]
        L1["TTFT (Time To First Token): Thời gian chờ ra chữ đầu tiên (Pha Prefill - Compute bound)"]
        L2["TPOT / ITL (Time Per Output Token): Tốc độ sinh từng chữ tiếp theo (Pha Decode - Memory bound)"]
        L1 --> L2
    end`, "Bản Đồ Quyết Định 2: Chọn Độ Đo Đánh Giá Chuẩn Xác (Module 5, 6, 7 & 11)");

            const diag3 = CustomDiagramEngine.renderBlock(`flowchart TD
    subgraph GDS_Layer["1. Đường truyền Lưu trữ -> GPU (GPUDirect Storage - GDS)"]
        NVMe["Ổ cứng Local NVMe SSD / Parallel FS (Lustre, WEKA)"] ==>|"GPUDirect Storage (GDS): DMA thẳng vào VRAM, bỏ qua CPU Bounce Buffer"| GPU1["GPU 1 VRAM (HBM3e - Node A)"]
    end
    subgraph Intra_Layer["2. Kết nối GPU Nội bộ Máy chủ (Intra-node: NVLink & NVSwitch)"]
        GPU_A1["GPU 1 VRAM (Node A)"] <==>|"NVLink / NVSwitch (900 GB/s - 1.8 TB/s): Dùng cho Tensor Parallelism"| GPU_A2["GPU 2 VRAM (Node A)"]
    end
    subgraph Inter_Layer["3. Kết nối GPU Giữa các Máy chủ (Inter-node: InfiniBand + GPUDirect RDMA)"]
        GPU_NodeA["GPU VRAM (Máy chủ Node A)"] <==>|"InfiniBand / Spectrum-X + GPUDirect RDMA: Truyền thẳng qua mạng giảm trung gian CPU"| GPU_NodeB["GPU VRAM (Máy chủ Node B)"]
    end
    GDS_Layer ==> Intra_Layer ==> Inter_Layer`, "Sơ Đồ 3: Kiến Trúc Truyền Dữ Liệu Tốc Độ Cao Giữa Storage & GPU (Module 1 & 8)");

            const diag4 = CustomDiagramEngine.renderBlock(`flowchart LR
    Raw["1. NVIDIA RAPIDS<br/>• cuIO -> cuDF (Pandas)<br/>• cuML (Sklearn)<br/>• cuGraph / cuVS"] ==> NeMo["2. NVIDIA NeMo<br/>• Curator (Lọc/Dedupe)<br/>• Megatron (Train 3D)<br/>• Customizer (LoRA/DPO)"] ==> TRT["3. TensorRT-LLM<br/>• Layer Fusion<br/>• FP8 / INT8 / INT4<br/>• PagedAttention"] ==> Triton["4. Triton Server<br/>• Đa Framework<br/>• Dynamic Batching<br/>• HTTP / gRPC"] ==> NIM["5. NVIDIA NIM & Safety<br/>• OpenAI-compatible API<br/>• NeMo Guardrails (Colang)<br/>• Confidential Computing"]`, "Sơ Đồ 4: Pipeline End-to-End Hệ Sinh Thái NVIDIA Full-Stack (Module 8 & 9)");

            container.innerHTML = `
                <div class="module-hero">
                    <div>
                        <span class="brand-badge">VISUAL ARCHITECTURE & DECISION TREES</span>
                        <h2 style="font-size:1.5rem; font-weight:800; margin-top:6px;">🏗️ Hệ Thống Sơ Đồ Quyết Định Phòng Thi & Máy Tính Nhẩm VRAM LLM</h2>
                        <p style="color:var(--text-secondary); margin-top:4px;">Các sơ đồ cây quyết định được thiết kế bằng giao diện Card trực quan giúp bạn định hướng đáp án đúng trong 10 giây.</p>
                    </div>
                </div>

                <!-- Interactive LLM VRAM Calculator -->
                <div class="hub-card" style="margin-bottom:28px;">
                    <h3>🧮 Máy Tính Dung Lượng VRAM GPU Cho LLM — Trọng Số, Công Thức KV Cache & Ngân Sách AdamW (Module 1, 6, 7, 8)</h3>
                    <p style="color:var(--text-secondary); font-size:0.9rem;">Thay đổi kích thước mô hình, chuẩn độ chính xác, cấu hình ngữ cảnh (<code>Batch Size × Seq Len</code>, <code>GQA vs. MHA</code>) và chế độ vận hành để thấy rõ sự khác biệt giữa Suy luận, PEFT (LoRA/QLoRA) và Full Fine-Tuning (<code>16N bytes/param</code> với AdamW):</p>
                    <div class="calc-controls">
                        <div class="calc-group">
                            <label>1. Kích thước Mô hình (Số tham số)</label>
                            <select class="calc-select" id="calcParams" onchange="updateVramCalc()">
                                <option value="8" selected>8B (VD: LLaMA-3 8B — 32 layers)</option>
                                <option value="13">13B (Mô hình cỡ trung — 40 layers)</option>
                                <option value="34">34B (CodeLLaMA 34B — 48 layers)</option>
                                <option value="70">70B (LLaMA-3 70B / Nemotron 70B — 80 layers)</option>
                                <option value="340">340B (Nemotron-4 340B — 96 layers)</option>
                                <option value="405">405B (LLaMA-3.1 405B — 126 layers)</option>
                            </select>
                        </div>
                        <div class="calc-group">
                            <label>2. Độ Chính Xác Trọng Số (Weight Precision)</label>
                            <select class="calc-select" id="calcPrecision" onchange="updateVramCalc()">
                                <option value="4">FP32 (32-bit = 4 bytes/param)</option>
                                <option value="2" selected>BF16 / FP16 (16-bit = 2 bytes/param)</option>
                                <option value="1">FP8 / INT8 (8-bit = 1 byte/param)</option>
                                <option value="0.5">INT4 / NF4 (4-bit = 0.5 byte/param)</option>
                            </select>
                        </div>
                        <div class="calc-group">
                            <label>3. Cấu hình KV Cache (Batch × SeqLen & GQA/MHA)</label>
                            <select class="calc-select" id="calcKvConfig" onchange="updateVramCalc()">
                                <option value="1|4096|8" selected>B=1, S=4,096 • GQA (8 KV heads)</option>
                                <option value="8|8192|8">B=8, S=8,192 • GQA (8 KV heads)</option>
                                <option value="8|8192|32">B=8, S=8,192 • MHA (32 KV heads)</option>
                                <option value="32|16384|8">B=32, S=16,384 • GQA (8 KV heads)</option>
                            </select>
                        </div>
                        <div class="calc-group">
                            <label>4. Chế độ Vận hành (Workload Mode)</label>
                            <select class="calc-select" id="calcMode" onchange="updateVramCalc()">
                                <option value="inference" selected>Suy luận (Weights + Công thức KV Cache + Activations)</option>
                                <option value="lora">PEFT LoRA / QLoRA (Frozen Base + Adapter States & Activations)</option>
                                <option value="full">Full Fine-Tuning Mixed Precision AdamW (~16N bytes + Activations)</option>
                            </select>
                        </div>
                    </div>
                    <div class="calc-result" id="vramCalcResult"></div>
                </div>

                ${diag1}
                ${diag2}
                ${diag3}
                ${diag4}
            `;

            updateVramCalc();
            await postProcessRenderedDOM(container);
            document.getElementById('mainScroll').scrollTop = 0;
        }

        function updateVramCalc() {
            const paramsEl = document.getElementById('calcParams');
            if (!paramsEl) return;
            const paramsB = parseFloat(paramsEl.value);
            const bytesPerParam = parseFloat(document.getElementById('calcPrecision').value);
            const kvCfg = (document.getElementById('calcKvConfig')?.value || '1|4096|8').split('|');
            const batchSize = parseInt(kvCfg[0], 10);
            const seqLen = parseInt(kvCfg[1], 10);
            const kvHeads = parseInt(kvCfg[2], 10);
            const mode = document.getElementById('calcMode').value;

            const layerMap = { 8: 32, 13: 40, 34: 48, 70: 80, 340: 96, 405: 126 };
            const numLayers = layerMap[paramsB] || 32;
            const headDim = 128;
            const kvBytes = bytesPerParam >= 2 ? 2 : 1; // BF16/FP16 KV cache (2B) or FP8/INT8 KV cache (1B)

            const weightGB = paramsB * bytesPerParam;
            let totalGB = weightGB;
            let breakdown = "";
            let gpuRec = "";

            if (mode === 'inference') {
                // KV Cache = 2 * B * S * L * n_kv_heads * d_head * b_kv (in bytes -> GiB)
                const kvCacheBytes = 2 * batchSize * seqLen * numLayers * kvHeads * headDim * kvBytes;
                const kvCacheGB = kvCacheBytes / (1024 ** 3);
                const actOverheadGB = Math.max(0.8, weightGB * 0.05);
                totalGB = weightGB + kvCacheGB + actOverheadGB;
                breakdown = `Trọng số (<code>${paramsB}B × ${bytesPerParam}B</code>): <strong>${weightGB.toFixed(1)} GB</strong> + KV Cache (<code>2×B=${batchSize}×S=${seqLen}×L=${numLayers}×H=${kvHeads}×128×${kvBytes}B</code>): <strong>${kvCacheGB.toFixed(2)} GB</strong> + Activations/CUDA Context: <strong>~${actOverheadGB.toFixed(1)} GB</strong>`;
            } else if (mode === 'lora') {
                const adapterAndActGB = Math.max(2.5, weightGB * 0.22 + (batchSize * seqLen / 4096) * 0.6);
                totalGB = weightGB + adapterAndActGB;
                breakdown = `Trọng số gốc đóng băng: <strong>${weightGB.toFixed(1)} GB</strong> + LoRA Adapter (<code>B×A</code>), Optimizer States của Adapter & Activations: <strong>~${adapterAndActGB.toFixed(1)} GB</strong>`;
            } else {
                // Mixed Precision Full Fine-Tuning with AdamW: 2N (BF16 weights) + 2N (BF16 grads) + 12N (FP32 master weights 4N + m 4N + v 4N) = 16N bytes static + activations
                const staticAdamGB = paramsB * 16;
                const actTrainGB = Math.max(4.0, paramsB * 2.0 * (batchSize / 1));
                totalGB = staticAdamGB + actTrainGB;
                breakdown = `Bộ nhớ tĩnh Mixed-Precision AdamW (<code>16N</code>): Trọng số BF16 (<strong>${(paramsB * 2).toFixed(0)} GB</strong>) + Đạo hàm BF16 (<strong>${(paramsB * 2).toFixed(0)} GB</strong>) + AdamW FP32 Master Weights, Momentum & Variance (<code>12N</code> = <strong>${(paramsB * 12).toFixed(0)} GB</strong>) + Activations: <strong>~${actTrainGB.toFixed(0)} GB</strong>`;
            }

            if (totalGB <= 24) {
                gpuRec = "✅ Chạy gọn trên <strong>1 GPU 24GB</strong> (RTX 4090 / L4) hoặc 1 phân vùng <strong>MIG</strong>.";
            } else if (totalGB <= 80) {
                gpuRec = "✅ Chạy vừa trên <strong>1 GPU NVIDIA A100 (80GB) hoặc H100 (80GB)</strong>!";
            } else if (totalGB <= 640) {
                const numGpus = Math.ceil(totalGB / 80);
                gpuRec = `⚡ Vượt quá 1 GPU 80GB! Cần tối thiểu <strong>${numGpus} GPU H100 80GB</strong> (VD: trong 1 node 8-GPU DGX H100 kết nối qua <strong>NVLink + Tensor Parallelism / ZeRO</strong>).`;
            } else {
                const numGpus = Math.ceil(totalGB / 80);
                gpuRec = `🚀 Quy mô Cụm Đa Máy Chủ! Cần <strong>${numGpus}+ GPU H100/B200</strong> (hoặc rack-scale <strong>GB200 NVL72</strong>) kết nối qua <strong>NVLink + InfiniBand (GPUDirect RDMA) + 3D Parallelism + ZeRO/FSDP</strong>.`;
            }

            document.getElementById('vramCalcResult').innerHTML = `
                <div>
                    <div style="font-size:0.8rem; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Tổng VRAM Ước Tính Cần Thiết</div>
                    <div class="calc-big-num">~${totalGB.toFixed(1)} GB VRAM</div>
                    <div style="font-size:0.86rem; color:var(--text-secondary); margin-top:4px;">${breakdown}</div>
                </div>
                <div style="max-width:440px; background:var(--bg-card); padding:12px 16px; border-radius:8px; border:1px solid var(--border-color); border-left:3.5px solid var(--nv-green); font-size:0.87rem; color:var(--text-secondary);">
                    ${gpuRec}
                </div>
            `;
        }

        // View 4: Flashcards View
        function renderFlashcardsView() {
            const container = document.getElementById('contentContainer');
            document.getElementById('sidebarRight').style.display = 'none';

            container.innerHTML = `
                <div class="module-hero">
                    <div>
                        <span class="brand-badge">ACTIVE RECALL FLASHCARDS</span>
                        <h2 style="font-size:1.5rem; font-weight:800; margin-top:6px;">⚡ Bộ Thẻ Flashcards Lật Mở Ôn Tập Phản Xạ Nhanh (${FLASHCARDS_DATA.length} Thẻ Cốt Lõi)</h2>
                        <p style="color:var(--text-secondary); margin-top:4px;">Bấm vào từng thẻ để lật xem đáp án. Tự trả lời trong đầu trước khi lật thẻ để ghi nhớ lâu gấp 3 lần!</p>
                    </div>
                    <button class="complete-btn" onclick="document.querySelectorAll('.flashcard').forEach(c => { c.classList.remove('flipped'); c.querySelector('.fc-answer').style.display='none'; })">🔄 Úp lại toàn bộ thẻ</button>
                </div>

                <div class="flashcard-grid">
                    ${FLASHCARDS_DATA.map((fc, idx) => `
                        <div class="flashcard" onclick="this.classList.toggle('flipped'); const a = this.querySelector('.fc-answer'); a.style.display = this.classList.contains('flipped') ? 'block' : 'none';">
                            <div>
                                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                                    <span class="brand-badge">${fc.cat}</span>
                                    <span style="font-size:0.75rem; color:var(--text-muted)">Thẻ #${idx + 1} • Bấm để lật</span>
                                </div>
                                <div style="font-weight:600; font-size:0.94rem; color:var(--text-primary); margin-bottom:12px;">❓ ${fc.q}</div>
                                <div class="fc-answer" style="font-size:0.85rem; color:var(--text-secondary); white-space:pre-line; padding-top:10px; border-top:1px dashed var(--border-color); display:none;">
                                    ${fc.a}
                                </div>
                            </div>
                            <div style="font-size:0.74rem; color:var(--text-muted); margin-top:12px; font-weight:500;">👆 Click để Hiện / Ẩn Đáp án</div>
                        </div>
                    `).join('')}
                </div>
            `;

            document.getElementById('mainScroll').scrollTop = 0;
        }

        // View 5: Interactive 60-Min Timed Exam (English & Vietnamese) + Instant Practice Modes
        let quizState = {
            mode: 'exam50_en',
            answered: {},
            flagged: {},
            submitted: false,
            score: 0,
            secondsLeft: 3600,
            timerRunning: false,
            timerId: null
        };

        function isStrictExamMode(mode) {
            return mode === 'exam50_en' || mode === 'exam50_vi' || mode === 'practice01';
        }

        function isEnglishQuizMode() {
            return quizState.mode === 'exam50_en' || quizState.mode === 'practice01';
        }

        function formatTimer(sec) {
            const m = Math.floor(sec / 60).toString().padStart(2, '0');
            const s = (sec % 60).toString().padStart(2, '0');
            return `${m}:${s}`;
        }

        window.toggleQuizTimer = function() {
            if (quizState.submitted) return;
            if (quizState.timerRunning) {
                clearInterval(quizState.timerId);
                quizState.timerRunning = false;
            } else {
                quizState.timerRunning = true;
                quizState.timerId = setInterval(() => {
                    if (quizState.secondsLeft > 0) {
                        quizState.secondsLeft--;
                        const el = document.getElementById('quizTimerDisplay');
                        if (el) el.textContent = formatTimer(quizState.secondsLeft);
                    } else {
                        clearInterval(quizState.timerId);
                        quizState.timerRunning = false;
                        if (isStrictExamMode(quizState.mode) && !quizState.submitted) {
                            submitStrictExam();
                        }
                    }
                }, 1000);
            }
            const btn = document.getElementById('quizTimerBtn');
            const isEn = isEnglishQuizMode();
            if (btn) btn.textContent = quizState.timerRunning ? (isEn ? '⏸️ Pause Timer' : '⏸️ Tạm dừng') : (isEn ? '▶️ Start 60m Timer' : '▶️ Bắt đầu giờ');
        };

        window.switchQuizMode = function(newMode) {
            if (quizState.timerId) clearInterval(quizState.timerId);
            quizState = {
                mode: newMode,
                answered: {},
                flagged: {},
                submitted: false,
                score: 0,
                secondsLeft: newMode === 'set30' ? 1800 : 3600,
                timerRunning: false,
                timerId: null
            };
            renderQuizView();
            document.getElementById('mainScroll').scrollTop = 0;
        };

        window.toggleFlagQuestion = function(qIdx) {
            quizState.flagged[qIdx] = !quizState.flagged[qIdx];
            const navBtn = document.getElementById(`qnav-${qIdx}`);
            const flagBtn = document.getElementById(`qflag-${qIdx}`);
            const isEn = isEnglishQuizMode();
            if (navBtn) {
                navBtn.classList.toggle('flagged', !!quizState.flagged[qIdx]);
            }
            if (flagBtn) {
                flagBtn.textContent = quizState.flagged[qIdx]
                    ? (isEn ? '🚩 Flagged for Review' : '🚩 Đã đánh dấu xem lại')
                    : (isEn ? '🏳️ Flag for Review' : '🏳️ Đánh dấu xem lại');
                flagBtn.style.borderColor = quizState.flagged[qIdx] ? 'var(--accent-amber)' : '';
                flagBtn.style.color = quizState.flagged[qIdx] ? 'var(--accent-amber)' : '';
            }
        };

        window.submitStrictExam = function() {
            if (quizState.submitted) return;
            if (quizState.timerId) clearInterval(quizState.timerId);
            quizState.timerRunning = false;
            quizState.submitted = true;

            const questions = parseMockQuestions(quizState.mode);
            let correctCount = 0;
            questions.forEach((q, idx) => {
                if (quizState.answered[idx] === q.answer) correctCount++;
            });
            quizState.score = correctCount;
            renderQuizView();
            document.getElementById('mainScroll').scrollTop = 0;
        };

        async function renderQuizView() {
            const container = document.getElementById('contentContainer');
            document.getElementById('sidebarRight').style.display = 'none';
            const questions = parseMockQuestions(quizState.mode);
            const strictMode = isStrictExamMode(quizState.mode);
            const isEn = isEnglishQuizMode();
            const revealAnswers = !strictMode || quizState.submitted;

            const total = questions.length;
            const answeredCount = Object.keys(quizState.answered).length;
            const modeTitles = {
                practice01: '📗 Practice Exam 01 (50 Questions • 60 Minutes • English)',
                exam50_en: '🇬🇧 Official English Timed Mock Exam (50 Questions • 60 Minutes • Submit to Reveal Score)',
                exam50_vi: '🇻🇳 Thi Thử Định Giờ Tiếng Việt (50 Câu • 60 Phút • Chỉ Hiện Đáp Án Khi Nộp Bài)',
                set50: '📘 Luyện Đề 50 Câu Chuẩn Blueprint (Hiện Đáp Án & Giải Thích Ngay)',
                set30: '⚡ Bộ 30 Câu Hỏi Luyện Tập Nhanh (Short Practice Test)',
                all80: '🔥 Tổng Hợp Toàn Bộ 80 Câu Hỏi Thực Chiến (50 + 30 Câu)'
            };
            const modeTitle = modeTitles[quizState.mode] || modeTitles.exam50_en;

            // Build Domain Breakdown if submitted in strict mode
            let scoreReportHtml = '';
            if (strictMode && quizState.submitted) {
                const pct = Math.round((quizState.score / total) * 100);
                const passed = pct >= 70;
                const timeUsedSec = 3600 - quizState.secondsLeft;
                const domainStats = {};
                questions.forEach((q, idx) => {
                    const dom = q.domain || 'General';
                    if (!domainStats[dom]) domainStats[dom] = { correct: 0, total: 0 };
                    domainStats[dom].total++;
                    if (quizState.answered[idx] === q.answer) domainStats[dom].correct++;
                });

                const domainRows = Object.entries(domainStats).map(([dom, st]) => {
                    const dpct = Math.round((st.correct / st.total) * 100);
                    return `
                        <tr>
                            <td><strong>${dom}</strong></td>
                            <td style="text-align:center;">${st.correct} / ${st.total}</td>
                            <td style="text-align:center; font-weight:700; color:${dpct >= 70 ? 'var(--nv-green-light)' : 'var(--accent-amber)'}">${dpct}%</td>
                        </tr>
                    `;
                }).join('');

                scoreReportHtml = `
                    <div class="hub-card" style="margin-bottom:20px; border-left:5px solid ${passed ? 'var(--nv-green)' : 'var(--accent-rose)'};">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px; margin-bottom:14px;">
                            <div>
                                <span class="brand-badge">${passed ? '✅ PASSED (ĐẠT CHUẨN >= 70%)' : '⚠️ NEEDS REVIEW (CẦN ÔN TẬP THÊM < 70%)'}</span>
                                <h3 style="font-size:1.35rem; margin-top:6px;">📊 ${isEn ? 'Mock Exam Score Report' : 'Báo Cáo Kết Quả Bài Thi Thử 50 Câu'}</h3>
                                <p style="color:var(--text-secondary); font-size:0.88rem;">
                                    ${isEn
                                        ? `Answered: <strong>${answeredCount}/${total}</strong> • Time Elapsed: <strong>${formatTimer(timeUsedSec)}</strong> • ${quizState.mode === 'practice01' ? 'Multiple-answer questions require an exact match (no partial credit).' : 'Answer Key Distribution: <code>A:12 | B:12 | C:13 | D:13</code>'}`
                                        : `Đã trả lời: <strong>${answeredCount}/${total}</strong> câu • Thời gian làm bài: <strong>${formatTimer(timeUsedSec)}</strong> • Phân bố đáp án chuẩn: <code>A:12 | B:12 | C:13 | D:13</code>`}
                                </p>
                            </div>
                            <div style="text-align:right;">
                                <div class="calc-big-num">${quizState.score} / ${total} (${pct}%)</div>
                            </div>
                        </div>
                        <div class="table-wrapper">
                            <table>
                                <thead>
                                    <tr>
                                        <th>${isEn ? 'Official Blueprint Domain' : 'Miền Kiến Thức Chính Thức (Blueprint Domain)'}</th>
                                        <th style="text-align:center;">${isEn ? 'Score' : 'Số Câu Đúng'}</th>
                                        <th style="text-align:center;">${isEn ? 'Accuracy' : 'Tỷ Lệ Đạt'}</th>
                                    </tr>
                                </thead>
                                <tbody>${domainRows}</tbody>
                            </table>
                        </div>
                    </div>
                `;
            }

            // Build Question Navigator Palette for 50-question modes
            const paletteHtml = `
                <div class="q-palette">
                    <div style="width:100%; display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; font-size:0.8rem; color:var(--text-secondary); flex-wrap:wrap; gap:8px;">
                        <span>🧭 <strong>${isEn ? 'Question Navigator (Click to jump):' : 'Bảng Điều Hướng Câu Hỏi (Bấm để nhảy tới câu):'}</strong></span>
                        <span>
                            ${strictMode && !quizState.submitted
                                ? (isEn ? '🟩 Answered • 🟧 Flagged for Review • ⬜ Unanswered (Answers hidden until Submit)' : '🟩 Đã chọn • 🟧 Đánh dấu xem lại • ⬜ Chưa làm (Ẩn đáp án cho tới khi Nộp bài)')
                                : (isEn ? '🟩 Correct • 🟥 Incorrect / Unanswered' : '🟩 Trả lời đúng • 🟥 Trả lời sai')}
                        </span>
                    </div>
                    ${questions.map((q, idx) => {
                        const chosen = quizState.answered[idx];
                        let cls = '';
                        if (revealAnswers && chosen) {
                            cls = chosen === q.answer ? 'graded-ok' : 'graded-bad';
                        } else if (chosen) {
                            cls = 'done';
                        }
                        if (quizState.flagged[idx] && !quizState.submitted) cls += ' flagged';
                        return `<button class="q-nav-btn ${cls}" id="qnav-${idx}" onclick="document.getElementById('qcard-${idx}')?.scrollIntoView({behavior:'smooth', block:'center'})">${idx + 1}</button>`;
                    }).join('')}
                </div>
            `;

            container.innerHTML = `
                <div class="module-hero">
                    <div>
                        <span class="brand-badge">${quizState.mode === 'practice01' ? 'INDEPENDENT PRACTICE MATERIAL • NOT OFFICIAL NVIDIA EXAM QUESTIONS' : 'INTERACTIVE MOCK EXAM • 5 OFFICIAL DOMAINS • BALANCED ANSWER KEY (A:12 / B:12 / C:13 / D:13)'}</span>
                        <h2 style="font-size:1.38rem; font-weight:800; margin-top:6px;">🎯 ${modeTitle}</h2>
                        <p style="color:var(--text-secondary); margin-top:4px; font-size:0.9rem;">
                            ${strictMode
                                ? (isEn
                                    ? 'Exam Simulation Mode: Select your answers (you can change choices anytime and flag items for review). Correct answers and detailed explanations are revealed ONLY after you click <strong>Submit Exam</strong>.'
                                    : 'Chế độ Phòng Thi Thật: Chọn hoặc đổi đáp án thoải mái, kết quả đúng/sai và lời giải chi tiết được ẩn hoàn toàn và CHỈ hiển thị sau khi bạn bấm <strong>Nộp Bài & Chấm Điểm</strong>.')
                                : 'Chế độ Luyện Tập Phản Xạ: Chọn đáp án A/B/C/D để xem ngay lập tức phân tích chi tiết tại sao đáp án đó đúng và tại sao các phương án nhiễu sai.'}
                        </p>
                        <div style="display:flex; gap:8px; margin-top:12px; flex-wrap:wrap;">
                            <button class="complete-btn ${quizState.mode === 'practice01' ? 'completed' : ''}" onclick="switchQuizMode('practice01')">📗 Practice Exam 01 (50Q • 60m • EN)</button>
                            <button class="complete-btn ${quizState.mode === 'exam50_en' ? 'completed' : ''}" onclick="switchQuizMode('exam50_en')">🇬🇧 English Timed Exam (50Q • 60m)</button>
                            <button class="complete-btn ${quizState.mode === 'exam50_vi' ? 'completed' : ''}" onclick="switchQuizMode('exam50_vi')">🇻🇳 Thi Thử 60 Phút TV (Nộp bài mới hiện điểm)</button>
                            <button class="complete-btn ${quizState.mode === 'set50' ? 'completed' : ''}" onclick="switchQuizMode('set50')">📘 Luyện 50 Câu (Hiện giải thích ngay)</button>
                            <button class="complete-btn ${quizState.mode === 'set30' ? 'completed' : ''}" onclick="switchQuizMode('set30')">⚡ Bộ 30 Câu Luyện Nhanh</button>
                            <button class="complete-btn ${quizState.mode === 'all80' ? 'completed' : ''}" onclick="switchQuizMode('all80')">🔥 Tất Cả 80 Câu</button>
                        </div>
                        ${quizState.mode === 'practice01' ? '<p style="color:var(--text-secondary); margin-top:10px;">46 single-answer questions and 4 multiple-answer questions. For multiple-answer questions, select all correct options; no partial credit. <a href="nca-genl-practice-01.md" download>Download Markdown</a> · <a href="nca-genl-practice-01.json" download>Download JSON</a></p>' : ''}
                    </div>
                    <div class="quiz-hero-controls">
                        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                            <div style="background:var(--bg-tertiary); padding:8px 14px; border-radius:8px; border:1px solid var(--border-color); font-weight:700; font-family:'JetBrains Mono', monospace;">
                                ⏱️ <span id="quizTimerDisplay" style="color:var(--text-primary)">${formatTimer(quizState.secondsLeft)}</span>
                            </div>
                            <button class="complete-btn" id="quizTimerBtn" onclick="toggleQuizTimer()">${quizState.timerRunning ? (isEn ? '⏸️ Pause' : '⏸️ Tạm dừng') : (isEn ? '▶️ Start 60m Timer' : '▶️ Bắt đầu giờ')}</button>
                        </div>
                        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                            <div style="background:var(--bg-tertiary); padding:8px 14px; border-radius:8px; border:1px solid var(--border-color); font-weight:700;">
                                ${strictMode && !quizState.submitted
                                    ? `${isEn ? 'Answered' : 'Đã làm'}: <span id="quizScoreText" style="color:var(--accent-blue)">${answeredCount} / ${total}</span>`
                                    : `${isEn ? 'Score' : 'Điểm'}: <span id="quizScoreText" style="color:var(--nv-green-light)">${quizState.score} / ${strictMode ? total : answeredCount}</span> (${total} Qs)`}
                            </div>
                            ${strictMode && !quizState.submitted
                                ? `<button class="complete-btn completed" onclick="submitStrictExam()">📤 ${isEn ? 'Submit Exam & Grade' : 'Nộp Bài & Chấm Điểm'}</button>`
                                : ''}
                            <button class="complete-btn" onclick="resetQuiz()">🔄 ${isEn ? 'Reset' : 'Làm lại'}</button>
                        </div>
                    </div>
                </div>

                ${scoreReportHtml}
                ${paletteHtml}

                <div id="quizQuestionsContainer">
                    ${questions.map((q, qIdx) => {
                        const chosen = quizState.answered[qIdx];
                        const isFlagged = !!quizState.flagged[qIdx];
                        return `
                            <div class="quiz-card" id="qcard-${qIdx}">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">
                                    <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                                        <span class="brand-badge">${q.title}</span>
                                        <span class="domain-pill">${q.domain}</span>
                                    </div>
                                    ${strictMode && !quizState.submitted ? `
                                        <button class="complete-btn" id="qflag-${qIdx}" style="padding:4px 10px; font-size:0.76rem; ${isFlagged ? 'border-color:var(--accent-amber); color:var(--accent-amber);' : ''}" onclick="toggleFlagQuestion(${qIdx})">
                                            ${isFlagged ? (isEn ? '🚩 Flagged for Review' : '🚩 Đã đánh dấu xem lại') : (isEn ? '🏳️ Flag for Review' : '🏳️ Đánh dấu xem lại')}
                                        </button>
                                    ` : ''}
                                </div>
                                <div style="font-size:0.98rem; font-weight:600; color:var(--text-primary); margin-bottom:14px;">${marked.parseInline(q.prompt)}</div>
                                <div>
                                    ${q.options.map(opt => {
                                        let cls = '';
                                        if (revealAnswers) {
                                            if (chosen) {
                                                if (q.answer.split(',').includes(opt.letter)) cls = 'correct';
                                                else if (chosen.split(',').includes(opt.letter)) cls = 'wrong';
                                            } else if (strictMode && quizState.submitted && q.answer.split(',').includes(opt.letter)) {
                                                cls = 'correct';
                                            }
                                        } else if (chosen && chosen.split(',').includes(opt.letter)) {
                                            cls = 'selected';
                                        }
                                        return `
                                            <button class="quiz-option ${cls}" data-letter="${opt.letter}" aria-pressed="${!!chosen && chosen.split(',').includes(opt.letter)}" onclick="answerQuizQuestion(${qIdx}, '${opt.letter}', '${q.answer}')">
                                                <strong style="color:var(--nv-green-light); min-width:24px;">${opt.letter}.</strong>
                                                <span>${marked.parseInline(opt.text)}</span>
                                            </button>
                                        `;
                                    }).join('')}
                                </div>
                                <div class="quiz-explanation ${(revealAnswers && (chosen || quizState.submitted)) ? 'visible' : ''}" id="qexp-${qIdx}">
                                    ${marked.parseInline(q.explanation)}
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>

                ${strictMode && !quizState.submitted ? `
                    <div style="display:flex; justify-content:center; margin-top:24px; margin-bottom:32px;">
                        <button class="complete-btn completed" style="padding:12px 28px; font-size:1rem;" onclick="submitStrictExam()">
                            📤 ${isEn ? 'Submit All 50 Answers & View Score Breakdown' : 'Nộp Bài Thi 50 Câu & Xem Điểm Theo 5 Miền Kiến Thức'}
                        </button>
                    </div>
                ` : ''}
            `;

            await postProcessRenderedDOM(container);
        }

        window.answerQuizQuestion = function(qIdx, selectedLetter, correctLetter) {
            const strictMode = isStrictExamMode(quizState.mode);
            if (strictMode) {
                if (quizState.submitted) return;
                // Automatically start the 60-minute timer on first answer selection in strict exam mode
                if (!quizState.timerRunning && Object.keys(quizState.answered).length === 0) {
                    toggleQuizTimer();
                }
                const question = parseMockQuestions(quizState.mode)[qIdx];
                let selected = selectedLetter;
                if (question.multiple) {
                    const letters = new Set((quizState.answered[qIdx] || '').split(',').filter(Boolean));
                    if (letters.has(selectedLetter)) letters.delete(selectedLetter);
                    else letters.add(selectedLetter);
                    selected = [...letters].sort().join(',');
                }
                if (selected) quizState.answered[qIdx] = selected;
                else delete quizState.answered[qIdx];
                const card = document.getElementById(`qcard-${qIdx}`);
                if (card) {
                    card.querySelectorAll('.quiz-option').forEach(btn => {
                        const active = selected.split(',').includes(btn.getAttribute('data-letter'));
                        btn.classList.toggle('selected', active);
                        btn.setAttribute('aria-pressed', String(active));
                    });
                }
                const navBtn = document.getElementById(`qnav-${qIdx}`);
                if (navBtn) navBtn.classList.toggle('done', !!selected);
                const answeredCount = Object.keys(quizState.answered).length;
                const scoreEl = document.getElementById('quizScoreText');
                if (scoreEl) scoreEl.textContent = `${answeredCount} / ${parseMockQuestions(quizState.mode).length}`;
                return;
            }

            if (quizState.answered[qIdx]) return;
            quizState.answered[qIdx] = selectedLetter;
            if (selectedLetter === correctLetter) {
                quizState.score++;
            }
            const card = document.getElementById(`qcard-${qIdx}`);
            const buttons = card.querySelectorAll('.quiz-option');
            buttons.forEach(btn => {
                const letter = btn.getAttribute('data-letter');
                if (letter === correctLetter) btn.classList.add('correct');
                else if (letter === selectedLetter && selectedLetter !== correctLetter) btn.classList.add('wrong');
            });
            document.getElementById(`qexp-${qIdx}`).classList.add('visible');
            const navBtn = document.getElementById(`qnav-${qIdx}`);
            if (navBtn) navBtn.classList.add(selectedLetter === correctLetter ? 'graded-ok' : 'graded-bad');
            const answeredCount = Object.keys(quizState.answered).length;
            document.getElementById('quizScoreText').textContent = `${quizState.score} / ${answeredCount}`;
        };

        window.resetQuiz = function() {
            switchQuizMode(quizState.mode);
        };

        function switchView(viewName, shouldRender = true) {
            currentView = viewName;
            document.querySelectorAll('#mainNavTabs .nav-tab').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.view === viewName);
            });
            closeMobileDrawers();
            syncMobileFabState();
            renderSidebar();

            if (!shouldRender) return;
            if (viewName === 'reader') renderReaderView();
            else if (viewName === 'mindmaps') renderMindmapsHubView();
            else if (viewName === 'diagrams') renderDiagramsHubView();
            else if (viewName === 'flashcards') renderFlashcardsView();
            else if (viewName === 'quiz') renderQuizView();
        }

        function performSearch(query) {
            const resultsEl = document.getElementById('searchResults');
            const q = query.trim().toLowerCase();
            if (q.length < 2) {
                resultsEl.innerHTML = `<div style="padding:20px; text-align:center; color:var(--text-muted);">Gõ ít nhất 2 ký tự để tìm kiếm tức thì trên toàn bộ 12 tài liệu...</div>`;
                return;
            }

            const matches = [];
            DOCS_DATA.forEach(doc => {
                const lines = doc.content.split(/\r?\n/);
                lines.forEach((line) => {
                    if (line.toLowerCase().includes(q)) {
                        const cleanLine = line.replace(/[#*>`]/g, '').trim();
                        if (cleanLine.length > 0) {
                            matches.push({
                                docId: doc.id,
                                badge: doc.badge,
                                title: doc.shortTitle,
                                snippet: cleanLine.substring(0, 180),
                                rawTerm: q
                            });
                        }
                    }
                });
            });

            if (matches.length === 0) {
                resultsEl.innerHTML = `<div style="padding:20px; text-align:center; color:var(--text-muted);">Không tìm thấy kết quả nào cho "<strong>${query}</strong>".</div>`;
                return;
            }

            const topMatches = matches.slice(0, 35);
            const regex = new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');

            resultsEl.innerHTML = topMatches.map(m => {
                const highlighted = m.snippet.replace(regex, '<mark>$1</mark>');
                return `
                    <div class="search-result-item" onclick="closeSearchModal(); selectDocument('${m.docId}', '${m.rawTerm.replace(/'/g, "\\'")}')">
                        <div style="font-size:0.75rem; font-weight:700; color:var(--nv-green-light); margin-bottom:4px;">${m.badge} — ${m.title}</div>
                        <div style="font-size:0.86rem; color:var(--text-secondary);">${highlighted}...</div>
                    </div>
                `;
            }).join('');
        }

        function openSearchModal() {
            document.getElementById('searchModal').classList.add('open');
            const input = document.getElementById('searchInput');
            input.focus();
            input.select();
            performSearch(input.value);
        }

        function closeSearchModal() {
            document.getElementById('searchModal').classList.remove('open');
        }

        document.getElementById('mainNavTabs').addEventListener('click', (e) => {
            const btn = e.target.closest('.nav-tab');
            if (btn) switchView(btn.dataset.view);
        });

        document.getElementById('toggleSidebarBtn').addEventListener('click', () => {
            const sidebarLeft = document.getElementById('sidebarLeft');
            const sidebarRight = document.getElementById('sidebarRight');
            const backdrop = document.getElementById('sidebarBackdrop');
            sidebarRight.classList.remove('mobile-open');
            const isNowCollapsed = sidebarLeft.classList.toggle('collapsed');
            if (window.innerWidth <= 860) {
                backdrop.classList.toggle('active', !isNowCollapsed);
            }
        });

        document.getElementById('sidebarBackdrop').addEventListener('click', () => {
            closeMobileDrawers();
        });

        document.getElementById('openTocFab').addEventListener('click', () => {
            const sidebarRight = document.getElementById('sidebarRight');
            const sidebarLeft = document.getElementById('sidebarLeft');
            const backdrop = document.getElementById('sidebarBackdrop');
            if (window.innerWidth <= 860) sidebarLeft.classList.add('collapsed');
            const isOpen = sidebarRight.classList.toggle('mobile-open');
            backdrop.classList.toggle('active', isOpen);
        });

        document.getElementById('closeTocMobileBtn').addEventListener('click', () => {
            document.getElementById('sidebarRight').classList.remove('mobile-open');
            document.getElementById('sidebarBackdrop').classList.remove('active');
        });

        window.addEventListener('resize', () => {
            syncMobileFabState();
            if (window.innerWidth > 860) {
                document.getElementById('sidebarBackdrop').classList.remove('active');
            }
        });

        document.getElementById('fontIncBtn').addEventListener('click', () => {
            fontSize = Math.min(22, fontSize + 1);
            document.body.style.setProperty('--base-font-size', `${fontSize}px`);
        });

        document.getElementById('fontDecBtn').addEventListener('click', () => {
            fontSize = Math.max(13, fontSize - 1);
            document.body.style.setProperty('--base-font-size', `${fontSize}px`);
        });

        document.getElementById('widthToggleBtn').addEventListener('click', () => {
            document.getElementById('contentContainer').classList.toggle('wide');
        });

        document.getElementById('themeToggleBtn').addEventListener('click', () => {
            const htmlEl = document.documentElement;
            const nextTheme = htmlEl.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            htmlEl.setAttribute('data-theme', nextTheme);
            document.getElementById('themeToggleBtn').textContent = nextTheme === 'dark' ? '☀️' : '🌙';
        });

        document.getElementById('openSearchBtn').addEventListener('click', openSearchModal);
        document.getElementById('closeSearchBtn').addEventListener('click', closeSearchModal);
        document.getElementById('searchModal').addEventListener('click', (e) => {
            if (e.target.id === 'searchModal') closeSearchModal();
        });
        document.getElementById('searchInput').addEventListener('input', (e) => performSearch(e.target.value));

        window.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
                e.preventDefault();
                openSearchModal();
            } else if (e.key === 'Escape') {
                closeSearchModal();
                closeMobileDrawers();
            }
        });

        window.addEventListener('DOMContentLoaded', () => {
            if (window.innerWidth <= 860) {
                document.getElementById('sidebarLeft').classList.add('collapsed');
            }
            syncMobileFabState();
            renderSidebar();
            renderReaderView();
        });
