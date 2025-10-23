/**
 * Grace AI Dashboard - Frontend Application
 * Communicates with the backend API and displays Grace's status in real-time
 */

const API_BASE = '/api';
let statusElement = document.getElementById('status-text');
let refreshInterval = null;

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('Grace Dashboard initialized');
    
    // Set up auto-refresh
    refreshStatus();
    refreshInterval = setInterval(refreshStatus, 2000);
    
    // Set up task form
    document.getElementById('task-form').addEventListener('submit', handleCreateTask);
    
    // Check API connectivity
    checkAPIConnection();
});

async function checkAPIConnection() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            statusElement.textContent = '✅ Connected to Grace';
            statusElement.style.color = 'green';
        } else {
            statusElement.textContent = '❌ API Error';
            statusElement.style.color = 'red';
        }
    } catch (error) {
        console.error('Connection error:', error);
        statusElement.textContent = '❌ Cannot connect to Grace';
        statusElement.style.color = 'red';
    }
}

async function refreshStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        if (!response.ok) throw new Error('API error');
        
        const data = await response.json();
        
        // Update KPIs
        document.getElementById('trust-score').textContent = (data.trust_score || 0).toFixed(1) + '%';
        document.getElementById('stability').textContent = (data.kpis.stability || 0).toFixed(1) + '%';
        document.getElementById('performance').textContent = (data.kpis.performance || 0).toFixed(1) + '%';
        document.getElementById('active-tasks').textContent = data.active_tasks || 0;
        
        // Update status
        statusElement.textContent = '✅ Connected to Grace';
        statusElement.style.color = 'green';
        
    } catch (error) {
        console.error('Error refreshing status:', error);
        statusElement.textContent = '❌ Connection lost';
        statusElement.style.color = 'red';
    }
}

async function loadTasks() {
    try {
        const response = await fetch(`${API_BASE}/tasks`);
        if (!response.ok) throw new Error('API error');
        
        const data = await response.json();
        const tasksList = document.getElementById('tasks-list');
        
        if (!data.tasks || data.tasks.length === 0) {
            tasksList.innerHTML = '<p>No tasks yet. Create one to get started!</p>';
            return;
        }
        
        tasksList.innerHTML = data.tasks.map(task => `
            <div class="task-item">
                <div class="title">${task.title}</div>
                <div class="status">Status: ${task.status}</div>
                <div class="status">Created by: ${task.created_by}</div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Error loading tasks:', error);
        document.getElementById('tasks-list').innerHTML = '<p>Error loading tasks</p>';
    }
}

async function handleCreateTask(event) {
    event.preventDefault();
    
    const title = document.getElementById('task-title').value;
    const description = document.getElementById('task-description').value;
    
    try {
        const response = await fetch(`${API_BASE}/tasks`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title: title,
                description: description
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('Task created:', data.task_id);
            
            // Clear form
            document.getElementById('task-form').reset();
            
            // Reload tasks
            await loadTasks();
            
            // Show success message
            alert('Task created successfully!');
        } else {
            alert('Error creating task');
        }
    } catch (error) {
        console.error('Error creating task:', error);
        alert('Error creating task: ' + error.message);
    }
}

// Load tasks on page load and refresh every 3 seconds
loadTasks();
setInterval(loadTasks, 3000);
