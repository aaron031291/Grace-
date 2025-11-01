/**
 * Grace Mobile App - React Native
 * 
 * Voice-first AI assistant in your pocket
 * 
 * Features:
 * - Voice interaction (speak to Grace)
 * - Real-time chat
 * - Code review on mobile
 * - Task management
 * - Knowledge upload
 * - Push notifications from Grace
 */

import React, { useState, useEffect } from 'react';
import {
  View, Text, TouchableOpacity, ScrollView, StyleSheet,
  ActivityIndicator, StatusBar
} from 'react-native';
import { Mic, Send, Brain, Bell } from 'react-native-feather';

export const GraceApp: React.FC = () => {
  const [messages, setMessages] = useState<any[]>([]);
  const [isListening, setIsListening] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [inputText, setInputText] = useState('');

  useEffect(() => {
    connectToGrace();
  }, []);

  const connectToGrace = async () => {
    // Connect to Grace WebSocket
    const ws = new WebSocket('wss://grace-api.yourdomain.com/api/grace/chat/ws');
    
    ws.onopen = () => {
      setIsConnected(true);
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleGraceMessage(data);
    };
  };

  const handleGraceMessage = (data: any) => {
    if (data.type === 'grace_notification') {
      // Show push notification
      showNotification(data.content);
    }
    
    addMessage({
      type: 'grace',
      content: data.content,
      timestamp: new Date()
    });
  };

  const startVoiceRecording = async () => {
    setIsListening(true);
    // Use react-native-voice or expo-speech
    // Record audio and send to Grace
  };

  const addMessage = (message: any) => {
    setMessages(prev => [...prev, message]);
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      {/* Header */}
      <View style={styles.header}>
        <Brain color="#fff" width={24} height={24} />
        <Text style={styles.headerTitle}>Grace AI</Text>
        <View style={[styles.statusDot, isConnected && styles.statusConnected]} />
      </View>

      {/* Messages */}
      <ScrollView style={styles.messages}>
        {messages.map((msg, idx) => (
          <View
            key={idx}
            style={[
              styles.messageBubble,
              msg.type === 'user' ? styles.userMessage : styles.graceMessage
            ]}
          >
            <Text style={styles.messageText}>{msg.content}</Text>
          </View>
        ))}
      </ScrollView>

      {/* Input */}
      <View style={styles.inputContainer}>
        <TouchableOpacity
          style={styles.voiceButton}
          onPress={startVoiceRecording}
        >
          <Mic color={isListening ? "#ef4444" : "#fff"} />
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.sendButton}>
          <Send color="#fff" />
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111827'
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#1f2937',
    gap: 12
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    flex: 1
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#ef4444'
  },
  statusConnected: {
    backgroundColor: '#10b981'
  },
  messages: {
    flex: 1,
    padding: 16
  },
  messageBubble: {
    padding: 12,
    borderRadius: 16,
    marginBottom: 8,
    maxWidth: '80%'
  },
  userMessage: {
    backgroundColor: '#3b82f6',
    alignSelf: 'flex-end'
  },
  graceMessage: {
    backgroundColor: '#374151',
    alignSelf: 'flex-start'
  },
  messageText: {
    color: '#fff',
    fontSize: 16
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
    backgroundColor: '#1f2937'
  },
  voiceButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#8b5cf6',
    justifyContent: 'center',
    alignItems: 'center'
  },
  sendButton: {
    flex: 1,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#8b5cf6',
    justifyContent: 'center',
    alignItems: 'center'
  }
});

export default GraceApp;
