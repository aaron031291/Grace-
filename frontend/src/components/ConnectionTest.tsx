/**
 * Connection Test Component
 * Tests connectivity to Grace backend services
 */

import React, { useState, useEffect } from 'react';

interface ConnectionStatus {
  backend: 'connected' | 'disconnected' | 'testing';
  websocket: 'connected' | 'disconnected' | 'testing';
  hunter: 'connected' | 'disconnected' | 'testing';
}

export const ConnectionTest: React.FC = () => {
  const [status, setStatus] = useState<ConnectionStatus>({
    backend: 'testing',
    websocket: 'testing',
    hunter: 'testing'
  });

  useEffect(() => {
    testConnections();
  }, []);

  const testConnections = async () => {
    // Test backend API
    try {
      const response = await fetch('http://localhost:8001/api/health');
      if (response.ok) {
        setStatus(prev => ({ ...prev, backend: 'connected' }));
      } else {
        setStatus(prev => ({ ...prev, backend: 'disconnected' }));
      }
    } catch {
      setStatus(prev => ({ ...prev, backend: 'disconnected' }));
    }

    // Test WebSocket
    try {
      const ws = new WebSocket('ws://localhost:8001/api/ws/orb');
      ws.onopen = () => {
        setStatus(prev => ({ ...prev, websocket: 'connected' }));
        ws.close();
      };
      ws.onerror = () => {
        setStatus(prev => ({ ...prev, websocket: 'disconnected' }));
      };
    } catch {
      setStatus(prev => ({ ...prev, websocket: 'disconnected' }));
    }

    // Test Hunter Protocol
    try {
      const response = await fetch('http://localhost:8001/api/hunter/stats');
      if (response.ok) {
        setStatus(prev => ({ ...prev, hunter: 'connected' }));
      } else {
        setStatus(prev => ({ ...prev, hunter: 'disconnected' }));
      }
    } catch {
      setStatus(prev => ({ ...prev, hunter: 'disconnected' }));
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return '#00ff88';
      case 'disconnected': return '#ff4444';
      case 'testing': return '#ffaa00';
      default: return '#999';
    }
  };

  return (
    <div style={{ padding: '2rem', background: '#1a1a2e', color: '#eee', borderRadius: '8px' }}>
      <h3>Connection Status</h3>
      <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ 
            width: '12px', 
            height: '12px', 
            borderRadius: '50%', 
            background: getStatusColor(status.backend) 
          }} />
          <span>Backend API: {status.backend}</span>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ 
            width: '12px', 
            height: '12px', 
            borderRadius: '50%', 
            background: getStatusColor(status.websocket) 
          }} />
          <span>WebSocket: {status.websocket}</span>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ 
            width: '12px', 
            height: '12px', 
            borderRadius: '50%', 
            background: getStatusColor(status.hunter) 
          }} />
          <span>Hunter Protocol: {status.hunter}</span>
        </div>
      </div>
      
      <button 
        onClick={testConnections}
        style={{
          marginTop: '1rem',
          padding: '0.5rem 1rem',
          background: '#667eea',
          border: 'none',
          borderRadius: '4px',
          color: 'white',
          cursor: 'pointer'
        }}
      >
        Retest
      </button>
    </div>
  );
};

export default ConnectionTest;
