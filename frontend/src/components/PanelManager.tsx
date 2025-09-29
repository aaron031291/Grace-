import React, { useState, useRef } from 'react'
import Draggable from 'react-draggable'
import { ResizableBox } from 'react-resizable'
import 'react-resizable/css/styles.css'

interface Panel {
  id: string
  type: string
  title: string
  x: number
  y: number
  width: number
  height: number
  zIndex: number
  isMinimized: boolean
  isMaximized: boolean
  config: any
}

export const PanelManager: React.FC = () => {
  const [panels, setPanels] = useState<Panel[]>([
    {
      id: '1',
      type: 'chat',
      title: 'Chat',
      x: 50,
      y: 50,
      width: 400,
      height: 300,
      zIndex: 1,
      isMinimized: false,
      isMaximized: false,
      config: {}
    },
    {
      id: '2',
      type: 'memory',
      title: 'Memory Explorer',
      x: 500,
      y: 100,
      width: 350,
      height: 400,
      zIndex: 2,
      isMinimized: false,
      isMaximized: false,
      config: {}
    }
  ])
  
  const [maxZIndex, setMaxZIndex] = useState(2)

  const bringToFront = (panelId: string) => {
    const newZIndex = maxZIndex + 1
    setMaxZIndex(newZIndex)
    setPanels(panels => 
      panels.map(panel =>
        panel.id === panelId ? { ...panel, zIndex: newZIndex } : panel
      )
    )
  }

  const updatePanel = (panelId: string, updates: Partial<Panel>) => {
    setPanels(panels =>
      panels.map(panel =>
        panel.id === panelId ? { ...panel, ...updates } : panel
      )
    )
  }

  const closePanel = (panelId: string) => {
    setPanels(panels => panels.filter(panel => panel.id !== panelId))
  }

  const minimizePanel = (panelId: string) => {
    updatePanel(panelId, { isMinimized: true })
  }

  const restorePanel = (panelId: string) => {
    updatePanel(panelId, { isMinimized: false, isMaximized: false })
  }

  const maximizePanel = (panelId: string) => {
    updatePanel(panelId, { isMaximized: true, isMinimized: false })
  }

  return (
    <div className="relative h-full w-full overflow-hidden">
      {panels.map(panel => (
        <PanelComponent
          key={panel.id}
          panel={panel}
          onBringToFront={() => bringToFront(panel.id)}
          onClose={() => closePanel(panel.id)}
          onMinimize={() => minimizePanel(panel.id)}
          onRestore={() => restorePanel(panel.id)}
          onMaximize={() => maximizePanel(panel.id)}
          onMove={(x, y) => updatePanel(panel.id, { x, y })}
          onResize={(width, height) => updatePanel(panel.id, { width, height })}
        />
      ))}
      
      {/* Panel toolbar */}
      <div className="absolute top-4 left-4 space-x-2">
        <button
          onClick={() => {
            const newPanel: Panel = {
              id: Date.now().toString(),
              type: 'chat',
              title: 'New Chat',
              x: Math.random() * 200 + 100,
              y: Math.random() * 200 + 100,
              width: 400,
              height: 300,
              zIndex: maxZIndex + 1,
              isMinimized: false,
              isMaximized: false,
              config: {}
            }
            setPanels([...panels, newPanel])
            setMaxZIndex(maxZIndex + 1)
          }}
          className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          + Chat
        </button>
        <button
          onClick={() => {
            const newPanel: Panel = {
              id: Date.now().toString(),
              type: 'memory',
              title: 'Memory Explorer',
              x: Math.random() * 200 + 200,
              y: Math.random() * 200 + 100,
              width: 350,
              height: 400,
              zIndex: maxZIndex + 1,
              isMinimized: false,
              isMaximized: false,
              config: {}
            }
            setPanels([...panels, newPanel])
            setMaxZIndex(maxZIndex + 1)
          }}
          className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700"
        >
          + Memory
        </button>
        <button
          onClick={() => {
            const newPanel: Panel = {
              id: Date.now().toString(),
              type: 'tasks',
              title: 'Task Board',
              x: Math.random() * 200 + 300,
              y: Math.random() * 200 + 100,
              width: 400,
              height: 350,
              zIndex: maxZIndex + 1,
              isMinimized: false,
              isMaximized: false,
              config: {}
            }
            setPanels([...panels, newPanel])
            setMaxZIndex(maxZIndex + 1)
          }}
          className="px-3 py-1 bg-purple-600 text-white rounded hover:bg-purple-700"
        >
          + Tasks
        </button>
      </div>
    </div>
  )
}

interface PanelComponentProps {
  panel: Panel
  onBringToFront: () => void
  onClose: () => void
  onMinimize: () => void
  onRestore: () => void
  onMaximize: () => void
  onMove: (x: number, y: number) => void
  onResize: (width: number, height: number) => void
}

const PanelComponent: React.FC<PanelComponentProps> = ({
  panel,
  onBringToFront,
  onClose,
  onMinimize,
  onRestore,
  onMaximize,
  onMove,
  onResize
}) => {
  const nodeRef = useRef<HTMLDivElement>(null)

  if (panel.isMinimized) {
    return (
      <div
        className="fixed bottom-4 left-4 bg-gray-800 border border-gray-600 rounded px-3 py-2 cursor-pointer"
        style={{ zIndex: panel.zIndex }}
        onClick={onRestore}
      >
        <span className="text-white text-sm">{panel.title}</span>
      </div>
    )
  }

  if (panel.isMaximized) {
    return (
      <div
        className="fixed inset-0 bg-gray-900 border border-gray-700"
        style={{ zIndex: panel.zIndex }}
        onClick={onBringToFront}
      >
        <div className="panel-header">
          <span className="text-white font-medium">{panel.title}</span>
          <div className="flex space-x-2">
            <button
              onClick={onRestore}
              className="w-6 h-6 bg-gray-600 hover:bg-gray-500 rounded flex items-center justify-center"
            >
              <span className="text-white text-xs">□</span>
            </button>
            <button
              onClick={onClose}
              className="w-6 h-6 bg-red-600 hover:bg-red-500 rounded flex items-center justify-center"
            >
              <span className="text-white text-xs">×</span>
            </button>
          </div>
        </div>
        <div className="panel-content">
          <PanelContent type={panel.type} config={panel.config} />
        </div>
      </div>
    )
  }

  return (
    <Draggable
      nodeRef={nodeRef}
      handle=".panel-header"
      position={{ x: panel.x, y: panel.y }}
      onStop={(e, data) => onMove(data.x, data.y)}
    >
      <div
        ref={nodeRef}
        className="absolute"
        style={{ zIndex: panel.zIndex }}
        onClick={onBringToFront}
      >
        <ResizableBox
          width={panel.width}
          height={panel.height}
          minConstraints={[200, 150]}
          maxConstraints={[800, 600]}
          onResize={(e, data) => onResize(data.size.width, data.size.height)}
          resizeHandles={['se']}
        >
          <div className="panel-container">
            <div className="panel-header">
              <span className="text-white font-medium">{panel.title}</span>
              <div className="flex space-x-2">
                <button
                  onClick={onMinimize}
                  className="w-6 h-6 bg-gray-600 hover:bg-gray-500 rounded flex items-center justify-center"
                >
                  <span className="text-white text-xs">−</span>
                </button>
                <button
                  onClick={onMaximize}
                  className="w-6 h-6 bg-gray-600 hover:bg-gray-500 rounded flex items-center justify-center"
                >
                  <span className="text-white text-xs">□</span>
                </button>
                <button
                  onClick={onClose}
                  className="w-6 h-6 bg-red-600 hover:bg-red-500 rounded flex items-center justify-center"
                >
                  <span className="text-white text-xs">×</span>
                </button>
              </div>
            </div>
            <div className="panel-content">
              <PanelContent type={panel.type} config={panel.config} />
            </div>
          </div>
        </ResizableBox>
      </div>
    </Draggable>
  )
}

interface PanelContentProps {
  type: string
  config: any
}

const PanelContent: React.FC<PanelContentProps> = ({ type, config }) => {
  switch (type) {
    case 'chat':
      return (
        <div className="h-full flex flex-col">
          <div className="flex-1 overflow-auto space-y-2 mb-4">
            <div className="bg-gray-700 rounded p-3">
              <div className="text-sm text-gray-300 mb-1">System</div>
              <div className="text-white">Welcome to Grace AI! How can I assist you today?</div>
            </div>
          </div>
          <div className="flex space-x-2">
            <input
              type="text"
              placeholder="Type your message..."
              className="flex-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white placeholder-gray-400"
            />
            <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
              Send
            </button>
          </div>
        </div>
      )
    
    case 'memory':
      return (
        <div className="h-full flex flex-col">
          <div className="mb-4">
            <input
              type="text"
              placeholder="Search memory fragments..."
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white placeholder-gray-400"
            />
          </div>
          <div className="flex-1 overflow-auto space-y-2">
            <div className="bg-gray-700 rounded p-3">
              <div className="text-sm text-green-400 mb-1">Document Fragment</div>
              <div className="text-white text-sm">Introduction to AI Governance principles...</div>
              <div className="text-xs text-gray-400 mt-2">Trust Score: 0.92</div>
            </div>
            <div className="bg-gray-700 rounded p-3">
              <div className="text-sm text-blue-400 mb-1">Chat Memory</div>
              <div className="text-white text-sm">Previous discussion about policy enforcement...</div>
              <div className="text-xs text-gray-400 mt-2">Trust Score: 0.87</div>
            </div>
          </div>
        </div>
      )
    
    case 'tasks':
      return (
        <div className="h-full">
          <div className="grid grid-cols-3 gap-4 h-full">
            <div>
              <h4 className="text-white font-medium mb-2">Pending</h4>
              <div className="space-y-2">
                <div className="bg-gray-700 rounded p-2">
                  <div className="text-white text-sm">Review policy changes</div>
                  <div className="text-xs text-gray-400">High priority</div>
                </div>
              </div>
            </div>
            <div>
              <h4 className="text-white font-medium mb-2">In Progress</h4>
              <div className="space-y-2">
                <div className="bg-gray-700 rounded p-2">
                  <div className="text-white text-sm">Update governance rules</div>
                  <div className="text-xs text-gray-400">50% complete</div>
                </div>
              </div>
            </div>
            <div>
              <h4 className="text-white font-medium mb-2">Completed</h4>
              <div className="space-y-2">
                <div className="bg-gray-700 rounded p-2">
                  <div className="text-white text-sm">Setup user permissions</div>
                  <div className="text-xs text-green-400">✓ Done</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )
    
    default:
      return (
        <div className="h-full flex items-center justify-center text-gray-400">
          Panel type: {type}
        </div>
      )
  }
}