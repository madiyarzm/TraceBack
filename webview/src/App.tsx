import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Node,
  Edge,
  BackgroundVariant,
  ReactFlowInstance,
} from '@xyflow/react';
import dagre from '@dagrejs/dagre';
import { toPng } from 'html-to-image';
import '@xyflow/react/dist/style.css';

import ToolNode, { ToolNodeData } from './nodes/ToolNode';
import ThinkingNode from './nodes/ThinkingNode';
import BatchNode, { BatchNodeData } from './nodes/BatchNode';
import SwimLaneNode, { SwimLanePersonality } from './nodes/SwimLaneNode';
import DetailDrawer, { DrawerPayload } from './components/DetailDrawer';
import EmptyState from './components/EmptyState';
import SessionPicker, { SessionSummary } from './components/SessionPicker';
import Toolbar from './components/Toolbar';
import SummaryBar from './components/SummaryBar';
import StumbleAlert, { StumbleAlertData } from './components/StumbleAlert';
import vscode from './vscodeApi';

const nodeTypes = {
  toolNode:     ToolNode,
  thinkingNode: ThinkingNode,
  batchNode:    BatchNode,
  swimLaneNode: SwimLaneNode,
};

const NODE_W      = 228;
const NODE_H      = 40;
const BATCH_W     = 244;
const THINKING_W  = 160;
const THINKING_H  = 26;

// Swimlane layout constants
const SWIM_PAD_X      = 36;
const SWIM_HEADER_H   = 28;
const SWIM_SUMMARY_H  = 20;
const SWIM_PAD_BOTTOM = 24;
const SWIM_GAP_Y      = 52;

const PERSONALITIES: SwimLanePersonality[] = [
  { name: 'Agent Smith', badge: '🕶️', color: '#00ff41', colorDim: 'rgba(0,255,65,0.04)'  },
  { name: 'Agent Brown', badge: '💼', color: '#cd7f32', colorDim: 'rgba(205,127,50,0.04)' },
  { name: 'Agent Jones', badge: '📁', color: '#6b8cff', colorDim: 'rgba(107,140,255,0.04)'},
];

type NodeStatus = 'pending' | 'success' | 'error' | 'thinking';

interface BatchItem {
  label:   string;
  detail?: string;
  status:  NodeStatus;
}

interface TraceNodeData {
  id:         string;
  toolName:   string;
  status:     NodeStatus;
  label:      string;
  count:      number;
  detail?:    string;
  toolInput?: Record<string, unknown>;
  isLooping?: boolean;
  timestamp:  number;
  isBatch?:   boolean;
  batchItems?: BatchItem[];
}

interface FullSessionData extends SessionSummary {
  nodes:        TraceNodeData[];
  stumbleAlert?: StumbleAlertData;
  aiSummary?:   string;
}

interface SessionUpdateMessage {
  type: 'session_update';
  session: {
    id:           string;
    nodes:        TraceNodeData[];
    stopped:      boolean;
    stumbleAlert?: StumbleAlertData;
    aiSummary?:   string;
  };
  allSessions: FullSessionData[];
}

interface LlmResponseMessage {
  type:   'llm_response';
  answer: string;
}

function getNodeType(tn: TraceNodeData): string {
  if (tn.toolName === '__thinking__') return 'thinkingNode';
  if (tn.isBatch) return 'batchNode';
  return 'toolNode';
}

function getNodeDimensions(type: string, itemCount = 0): { w: number; h: number } {
  if (type === 'thinkingNode') return { w: THINKING_W, h: THINKING_H };
  if (type === 'batchNode')    return { w: BATCH_W,    h: 56 + Math.min(itemCount, 3) * 22 };
  return { w: NODE_W, h: NODE_H };
}

function buildNodeData(tn: TraceNodeData, type: string): Record<string, unknown> {
  if (type === 'thinkingNode') return {};
  if (type === 'batchNode') {
    return {
      toolName:   tn.toolName,
      status:     tn.status,
      count:      tn.count,
      detail:     tn.detail,
      timestamp:  tn.timestamp,
      isBatch:    true,
      batchItems: tn.batchItems ?? [],
    } satisfies Omit<BatchNodeData, '[key: string]'>;
  }
  return {
    toolName:  tn.toolName,
    label:     tn.label,
    status:    tn.status,
    count:     tn.count,
    detail:    tn.detail,
    toolInput: tn.toolInput,
    isLooping: tn.isLooping,
    timestamp: tn.timestamp,
  } satisfies Omit<ToolNodeData, '[key: string]'>;
}

function buildEdges(traceNodes: TraceNodeData[], prefix = ''): Edge[] {
  return traceNodes.slice(1).map((tn, i) => {
    const isThinking = traceNodes[i].toolName === '__thinking__';
    return {
      id:     `${prefix}e-${traceNodes[i].id}-${tn.id}`,
      source: traceNodes[i].id,
      target: tn.id,
      style: {
        stroke:          'var(--tb-border-2, #30363d)',
        strokeWidth:     1.5,
        strokeDasharray: isThinking ? '4 3' : undefined,
      },
      animated: traceNodes[i].status === 'pending' || isThinking,
    };
  });
}

const NODE_GAP_TB = 24;

function buildReactFlowGraph(traceNodes: TraceNodeData[]): { nodes: Node[]; edges: Edge[] } {
  let yOffset = 0;
  const rfNodes: Node[] = traceNodes.map((tn) => {
    const type = getNodeType(tn);
    const data = buildNodeData(tn, type);
    const itemCount = (data.batchItems as unknown[] | undefined)?.length ?? 0;
    const { w, h } = getNodeDimensions(type, itemCount);
    const gap = type === 'thinkingNode' ? 12 : NODE_GAP_TB;
    const node: Node = { id: tn.id, type, position: { x: -w / 2, y: yOffset }, data };
    yOffset += h + gap;
    return node;
  });
  return { nodes: rfNodes, edges: buildEdges(traceNodes) };
}

function applyDagreLayout(
  rfNodes: Node[],
  rfEdges: Edge[],
): { positioned: Node[]; totalW: number; totalH: number } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'LR', ranksep: 40, nodesep: 20 });

  for (const node of rfNodes) {
    const itemCount = (node.data?.batchItems as unknown[] | undefined)?.length ?? 0;
    const { w, h } = getNodeDimensions(node.type ?? 'toolNode', itemCount);
    g.setNode(node.id, { width: w, height: h });
  }
  for (const edge of rfEdges) g.setEdge(edge.source, edge.target);

  dagre.layout(g);

  const gData = g.graph() as { width?: number; height?: number };
  const totalW = gData.width  ?? 400;
  const totalH = gData.height ?? 80;

  const positioned = rfNodes.map((node) => {
    const pos = g.node(node.id);
    const itemCount = (node.data?.batchItems as unknown[] | undefined)?.length ?? 0;
    const { w, h } = getNodeDimensions(node.type ?? 'toolNode', itemCount);
    const x = pos && Number.isFinite(pos.x) ? pos.x - w / 2 : 0;
    const y = pos && Number.isFinite(pos.y) ? pos.y - h / 2 : 0;
    return { ...node, position: { x, y } };
  });

  return { positioned, totalW, totalH };
}

function layoutSessionLR(sessionNodes: TraceNodeData[]): {
  laidOutNodes: Node[];
  edges:        Edge[];
  totalW:       number;
  totalH:       number;
} {
  if (sessionNodes.length === 0) {
    return { laidOutNodes: [], edges: [], totalW: 220, totalH: 60 };
  }
  const rfNodes: Node[] = sessionNodes.map((tn) => {
    const type = getNodeType(tn);
    return { id: tn.id, type, position: { x: 0, y: 0 }, data: buildNodeData(tn, type) };
  });
  const rfEdges = buildEdges(sessionNodes);
  const { positioned, totalW, totalH } = applyDagreLayout(rfNodes, rfEdges);
  return { laidOutNodes: positioned, edges: rfEdges, totalW, totalH };
}

function buildSwimlaneGraph(sessions: FullSessionData[]): { nodes: Node[]; edges: Edge[] } {
  const allNodes: Node[] = [];
  const allEdges: Edge[] = [];
  let yOffset = 0;

  sessions.forEach((session, idx) => {
    const personality = PERSONALITIES[idx % PERSONALITIES.length];
    const { laidOutNodes, edges, totalW, totalH } = layoutSessionLR(session.nodes);

    const headerH = SWIM_HEADER_H + (session.aiSummary ? SWIM_SUMMARY_H : 0);
    const laneW   = Math.max(totalW + SWIM_PAD_X * 2, 320);
    const laneH   = headerH + totalH + SWIM_PAD_BOTTOM;
    const laneId  = `swimlane-${session.id}`;

    allNodes.push({
      id:       laneId,
      type:     'swimLaneNode',
      position: { x: 0, y: yOffset },
      style:    { width: laneW, height: laneH },
      data: {
        label:      session.label,
        personality,
        nodeCount:  session.nodes.filter((n) => n.toolName !== '__thinking__').length,
        stopped:    session.stopped,
        aiSummary:  session.aiSummary,
      },
      draggable:  false,
      selectable: false,
    });

    const childNodes = laidOutNodes.map((node) => ({
      ...node,
      parentId: laneId,
      extent:   'parent' as const,
      position: {
        x: node.position.x + SWIM_PAD_X,
        y: node.position.y + headerH + 4,
      },
      draggable: false,
    }));

    allNodes.push(...childNodes);
    allEdges.push(...edges);

    yOffset += laneH + SWIM_GAP_Y;
  });

  return { nodes: allNodes, edges: allEdges };
}

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [hasData, setHasData]               = useState(false);
  const [isLive, setIsLive]                 = useState(false);
  const [actionCount, setActionCount]       = useState(0);
  const [drawer, setDrawer]                 = useState<DrawerPayload | null>(null);
  const [sessions, setSessions]             = useState<FullSessionData[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [currentSession, setCurrentSession] = useState<SessionUpdateMessage['session'] | null>(null);
  const [stumbleAlert, setStumbleAlert]     = useState<StumbleAlertData | null>(null);
  const [traceNodes, setTraceNodes]         = useState<TraceNodeData[]>([]);
  const [aiSummary, setAiSummary]           = useState<string | undefined>(undefined);
  const [chatAnswer, setChatAnswer]         = useState<string | undefined>(undefined);
  const [chatLoading, setChatLoading]       = useState(false);

  const rfInstanceRef    = useRef<ReactFlowInstance | null>(null);
  const flowContainerRef = useRef<HTMLDivElement>(null);

  const swimlaneMode = sessions.length > 1;

  useEffect(() => {
    function handleMessage(event: MessageEvent) {
      const message = event.data as SessionUpdateMessage | LlmResponseMessage;

      if (message.type === 'llm_response') {
        setChatAnswer((message as LlmResponseMessage).answer);
        setChatLoading(false);
        return;
      }

      if (message.type !== 'session_update') return;
      const msg = message as SessionUpdateMessage;

      const sessionNodes = msg.session.nodes;
      const realNodes    = sessionNodes.filter((n) => n.toolName !== '__thinking__');

      setCurrentSession(msg.session);
      setActiveSessionId(msg.session.id);
      setSessions(msg.allSessions ?? []);
      setHasData(sessionNodes.length > 0);
      setIsLive(!msg.session.stopped);
      setActionCount(realNodes.length);
      setTraceNodes(sessionNodes);
      setAiSummary(msg.session.aiSummary);

      if (msg.session.stumbleAlert) setStumbleAlert(msg.session.stumbleAlert);

      const isSwimlane = (msg.allSessions?.length ?? 0) > 1;
      if (isSwimlane) {
        const { nodes: rfNodes, edges: rfEdges } = buildSwimlaneGraph(msg.allSessions);
        setNodes(rfNodes);
        setEdges(rfEdges);
      } else {
        const { nodes: rfNodes, edges: rfEdges } = buildReactFlowGraph(sessionNodes);
        setNodes(rfNodes);
        setEdges(rfEdges);
      }

      setTimeout(() => rfInstanceRef.current?.fitView({ padding: 0.3, duration: 300 }), 50);
    }

    window.addEventListener('message', handleMessage);
    vscode.postMessage({ type: 'ready' });
    return () => window.removeEventListener('message', handleMessage);
  }, [setNodes, setEdges]);

  const onConnect = useCallback(
    (connection: Connection) => setEdges((eds) => addEdge(connection, eds)),
    [setEdges]
  );

  function handleNodeClick(_: React.MouseEvent, node: Node) {
    if (node.type === 'thinkingNode' || node.type === 'swimLaneNode') return;

    setChatAnswer(undefined);

    if (node.type === 'batchNode') {
      const d = node.data as BatchNodeData;
      setDrawer({
        toolName:   d.toolName,
        label:      `${d.count} steps`,
        status:     d.status,
        detail:     d.detail,
        isBatch:    true,
        batchItems: d.batchItems,
      });
      return;
    }

    const d = node.data as ToolNodeData;
    setDrawer({
      toolName:  d.toolName,
      label:     d.label,
      status:    d.status,
      detail:    d.detail,
      toolInput: d.toolInput,
    });
  }

  function handleChat(question: string) {
    setChatLoading(true);
    const nodeContext = traceNodes
      .filter((n) => n.toolName !== '__thinking__')
      .map((n) => `${n.toolName}: ${n.label} [${n.status}]`)
      .join('\n');
    vscode.postMessage({ type: 'llm_query', question, nodeContext });
  }

  async function handleExportPng() {
    const container = flowContainerRef.current;
    if (!container) return;
    try {
      const url = await toPng(container, { backgroundColor: '#07090d', pixelRatio: 2 });
      const a = document.createElement('a');
      a.download = `traceback-${Date.now()}.png`;
      a.href = url;
      a.click();
    } catch (err) {
      console.error('PNG export failed:', err);
    }
  }

  function handleExportJson() {
    if (!currentSession) return;
    const blob = new Blob([JSON.stringify(currentSession, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.download = `traceback-${Date.now()}.json`;
    a.href = url;
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleClear() {
    vscode.postMessage({ type: 'clear_session' });
    setNodes([]);
    setEdges([]);
    setHasData(false);
    setDrawer(null);
    setStumbleAlert(null);
    setTraceNodes([]);
    setAiSummary(undefined);
    setChatAnswer(undefined);
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100vw', height: '100vh' }}>
      {/* SessionPicker hidden in swimlane mode — swimlanes show all sessions */}
      {!swimlaneMode && (
        <SessionPicker sessions={sessions} activeId={activeSessionId} onSelect={setActiveSessionId} />
      )}

      {hasData && (
        <Toolbar
          isLive={isLive}
          nodeCount={actionCount}
          onExportPng={handleExportPng}
          onExportJson={handleExportJson}
          onClear={handleClear}
        />
      )}

      {hasData && !swimlaneMode && (
        <SummaryBar nodes={traceNodes} aiSummary={aiSummary} isLive={isLive} />
      )}

      <div style={{ position: 'relative', flex: 1, minHeight: 0 }} ref={flowContainerRef}>
        {!hasData ? (
          <EmptyState />
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={handleNodeClick}
            onPaneClick={() => setDrawer(null)}
            onInit={(instance) => { rfInstanceRef.current = instance; }}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.3 }}
            minZoom={0.2}
            maxZoom={2}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
            <Controls showInteractive={false} />
            <MiniMap
              nodeColor={(node) => {
                if (node.type === 'thinkingNode' || node.type === 'swimLaneNode') return '#58a6ff';
                const d = node.data as { status?: string };
                switch (d.status) {
                  case 'success': return '#3fb950';
                  case 'error':   return '#f85149';
                  case 'pending': return '#d29922';
                  default:        return '#58a6ff';
                }
              }}
              maskColor="rgba(7,9,13,0.5)"
            />
          </ReactFlow>
        )}

        {/* Stumble alert red-tint canvas overlay */}
        {stumbleAlert && (
          <div style={{
            position: 'absolute', inset: 0,
            background: 'radial-gradient(ellipse at center, rgba(248,81,73,0.07) 0%, rgba(248,81,73,0.02) 60%, transparent 100%)',
            pointerEvents: 'none',
            zIndex: 10,
          }} />
        )}

        <StumbleAlert alert={stumbleAlert} onDismiss={() => setStumbleAlert(null)} />
        <DetailDrawer
          node={drawer}
          onClose={() => setDrawer(null)}
          onChat={handleChat}
          chatAnswer={chatAnswer}
          chatLoading={chatLoading}
        />
      </div>
    </div>
  );
}
