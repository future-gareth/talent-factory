"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  Brain, 
  Database, 
  Zap, 
  Cpu, 
  Settings, 
  Plus, 
  Play, 
  CheckCircle, 
  AlertCircle,
  Upload,
  Download,
  X,
  Square
} from "lucide-react"

interface Dashboard {
  talent_count: number
  dataset_count: number
  active_training: number
  recent_runs: Array<{
    id: string
    base_model: string
    status: string
    created_at: string
  }>
  environment: {
    gpu_name?: string
    vram_gb?: number
    cpu_cores: number
    ram_gb: number
    ready: boolean
  }
}

interface Model {
  id: string
  name: string
  size_gb: number
  min_vram_gb: number
  description: string
  category: string
}

interface Talent {
  id: string
  name: string
  category?: string
  version?: string
  status: string
  created_at: string
  metrics: Record<string, any>
}

export default function TalentFactory() {
  const [currentView, setCurrentView] = useState<'dashboard' | 'wizard' | 'catalogue' | 'settings'>('dashboard')
  const [dashboard, setDashboard] = useState<Dashboard | null>(null)
  const [models, setModels] = useState<Model[]>([])
  const [talents, setTalents] = useState<Talent[]>([])
  const [loading, setLoading] = useState(true)

  // Wizard state
  const [wizardStep, setWizardStep] = useState(1)
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([])
  const [outcomePreference, setOutcomePreference] = useState<'speed' | 'balanced' | 'quality'>('balanced')
  const [trainingStatus, setTrainingStatus] = useState<any>(null)
  const [evaluationResults, setEvaluationResults] = useState<any>(null)
  const [talentName, setTalentName] = useState('')
  const [talentCategory, setTalentCategory] = useState('general')
  const [activeTrainingId, setActiveTrainingId] = useState<string | null>(null)
  const [websocket, setWebsocket] = useState<WebSocket | null>(null)
  
  // Hugging Face dataset search
  const [hfSearchQuery, setHfSearchQuery] = useState('')
  const [hfSearchResults, setHfSearchResults] = useState<any[]>([])
  const [hfSearchLoading, setHfSearchLoading] = useState(false)
  const [addedDatasets, setAddedDatasets] = useState<Set<string>>(new Set())
  
  // Existing datasets
  const [existingDatasets, setExistingDatasets] = useState<any[]>([])
  const [existingDatasetsLoading, setExistingDatasetsLoading] = useState(false)
  

  useEffect(() => {
    loadData()
    checkForActiveTraining()
    initializeWebSocket()
    
    // Clean up any stale training IDs on mount
    const cleanupStaleTraining = async () => {
      const savedTrainingId = localStorage.getItem('activeTrainingId')
      if (savedTrainingId) {
        try {
          const response = await fetch(`http://localhost:8084/train/status/${savedTrainingId}`)
          if (!response.ok) {
            console.log(`Cleaning up stale training ID: ${savedTrainingId}`)
            localStorage.removeItem('activeTrainingId')
          }
        } catch (error) {
          console.log(`Error checking training status, clearing localStorage: ${error}`)
          localStorage.removeItem('activeTrainingId')
        }
      }
    }
    
    cleanupStaleTraining()
    
    return () => {
      if (websocket) {
        websocket.close()
      }
    }
  }, [])

  // Poll for training status if WebSocket connection is lost or no progress updates
  useEffect(() => {
    if (activeTrainingId) {
      const initialProgress = (trainingStatus && typeof trainingStatus.progress === 'number') ? trainingStatus.progress : 0
      let lastProgress = initialProgress
      
      const pollInterval = setInterval(async () => {
        try {
          const response = await fetch(`http://localhost:8084/train/status/${activeTrainingId}`)
          if (response.ok) {
            const status = await response.json()
            console.log('Polling training status:', status.status, 'progress:', status.progress)
            
            // Update status if it's different
            const nextProgress = (typeof status.progress === 'number') ? status.progress : 0
            const prevStatus = trainingStatus ? trainingStatus.status : undefined
            if (nextProgress !== lastProgress || status.status !== prevStatus) {
              console.log('Progress changed, updating UI')
              setTrainingStatus(status)
              lastProgress = nextProgress
            }
            
            if (status.status === 'completed' || status.status === 'failed' || status.status === 'stopped') {
              console.log('Training finished, clearing state')
              localStorage.removeItem('activeTrainingId')
              setActiveTrainingId(null)
              clearInterval(pollInterval)
              
              if (status.status === 'completed') {
                runEvaluation(activeTrainingId)
              }
            }
          } else if (response.status === 404) {
            // Training not found, clear stale training ID
            console.log('Training not found (404), clearing stale training ID')
            localStorage.removeItem('activeTrainingId')
            setActiveTrainingId(null)
            clearInterval(pollInterval)
          }
        } catch (error) {
          console.error('Error polling training status:', error)
          
          // If we get a connection error or 404, the training might not exist anymore
          // Clear the stale training ID to stop polling
          if (error.message && (error.message.includes('Failed to fetch') || error.message.includes('404'))) {
            console.log('Training not found, clearing stale training ID')
            localStorage.removeItem('activeTrainingId')
            setActiveTrainingId(null)
            clearInterval(pollInterval)
          }
        }
      }, 3000) // Poll every 3 seconds
      
      return () => clearInterval(pollInterval)
    }
  }, [activeTrainingId, trainingStatus])

  const initializeWebSocket = () => {
    try {
      const ws = new WebSocket('ws://localhost:8084/ws')
      
      ws.onopen = () => {
        console.log('WebSocket connected')
        setWebsocket(ws)
      }
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          console.log('WebSocket message received:', message)
          handleWebSocketMessage(message)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }
      
      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setWebsocket(null)
        // Attempt to reconnect after 3 seconds
        setTimeout(() => {
          if (!websocket) {
            initializeWebSocket()
          }
        }, 3000)
      }
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error)
    }
  }

  const handleWebSocketMessage = (message: any) => {
    console.log('Handling WebSocket message:', message.type, 'for training:', message.train_id, 'active:', activeTrainingId)
    switch (message.type) {
      case 'training_update':
        if (message.train_id === activeTrainingId) {
          console.log('Updating training status:', message.data)
          setTrainingStatus({
            ...message.data,
            train_id: message.train_id
          })
          
          // Handle completion
          if (message.data.status === 'completed') {
            console.log('Training completed, clearing localStorage and running evaluation')
            localStorage.removeItem('activeTrainingId')
            setActiveTrainingId(null)
            runEvaluation(message.train_id)
          } else if (message.data.status === 'failed' || message.data.status === 'stopped') {
            console.log('Training failed/stopped, clearing localStorage')
            localStorage.removeItem('activeTrainingId')
            setActiveTrainingId(null)
          }
        } else {
          console.log('Message for different training ID, ignoring')
        }
        break
        
      case 'subscription_confirmed':
        console.log(`Subscribed to training ${message.train_id}: ${message.status}`)
        break
        
      case 'pong':
        // Handle ping/pong for connection health
        break
        
      default:
        console.log('Unknown WebSocket message:', message)
    }
  }

  const subscribeToTraining = (trainId: string) => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      websocket.send(JSON.stringify({
        type: 'subscribe_training',
        train_id: trainId
      }))
    }
  }

  const unsubscribeFromTraining = (trainId: string) => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      websocket.send(JSON.stringify({
        type: 'unsubscribe_training',
        train_id: trainId
      }))
    }
  }

  const checkForActiveTraining = async () => {
    try {
      // First check localStorage
      const savedTrainingId = localStorage.getItem('activeTrainingId')
      if (savedTrainingId) {
        const response = await fetch(`http://localhost:8084/train/status/${savedTrainingId}`)
        if (response.ok) {
          const status = await response.json()
          if (status.status === 'running' || status.status === 'preparing') {
            setActiveTrainingId(savedTrainingId)
            setTrainingStatus(status)
            setWizardStep(3) // Go to training step
            
            // Subscribe to WebSocket updates instead of polling
            setTimeout(() => {
              subscribeToTraining(savedTrainingId)
            }, 1000) // Wait for WebSocket to be ready
            return
          } else {
            // Training completed or failed, clear from localStorage
            localStorage.removeItem('activeTrainingId')
          }
        } else {
          // Training not found (404), clear from localStorage
          console.log(`Clearing stale training ID from localStorage: ${savedTrainingId}`)
          localStorage.removeItem('activeTrainingId')
        }
      }
      
      // Also check for any active trainings from backend
      const activeResponse = await fetch('http://localhost:8084/train/active')
      if (activeResponse.ok) {
        const activeData = await activeResponse.json()
        if (activeData.count > 0 && activeData.active_trainings.length > 0) {
          const activeTrainingId = activeData.active_trainings[0] // Take the first one
          const statusResponse = await fetch(`http://localhost:8084/train/status/${activeTrainingId}`)
          if (statusResponse.ok) {
            const status = await statusResponse.json()
            if (status.status === 'running' || status.status === 'preparing') {
              setActiveTrainingId(activeTrainingId)
              setTrainingStatus(status)
              setWizardStep(3) // Go to training step
              
              // Save to localStorage for persistence
              localStorage.setItem('activeTrainingId', activeTrainingId)
              
              // Subscribe to WebSocket updates
              setTimeout(() => {
                subscribeToTraining(activeTrainingId)
              }, 1000) // Wait for WebSocket to be ready
            }
          }
        }
      }
    } catch (error) {
      console.error('Failed to check for active training:', error)
    }
  }

  const loadExistingDatasets = async () => {
    try {
      setExistingDatasetsLoading(true)
      const response = await fetch('http://localhost:8084/datasets/available')
      if (response.ok) {
        const data = await response.json()
        setExistingDatasets(data.datasets || [])
      }
    } catch (error) {
      console.error('Failed to load existing datasets:', error)
    } finally {
      setExistingDatasetsLoading(false)
    }
  }

  const loadData = async () => {
    try {
      // Set default data first to show UI
      setDashboard({
        talent_count: 0,
        dataset_count: 0,
        active_training: 0,
        recent_runs: [],
        environment: {
          gpu_name: null,
          vram_gb: null,
          cpu_cores: 10,
          ram_gb: 16.0,
          ready: true
        }
      })
      
      setModels([
        {
          id: 'llama-2-7b',
          name: 'Llama 2 7B',
          size_gb: 13,
          min_vram_gb: 8,
          description: 'Efficient 7B parameter model',
          category: 'general'
        },
        {
          id: 'mistral-7b',
          name: 'Mistral 7B',
          size_gb: 14,
          min_vram_gb: 8,
          description: 'High-quality 7B parameter model',
          category: 'general'
        }
      ])
      
      setTalents([])
      
      // Load existing datasets
      await loadExistingDatasets()
      
      // Try to fetch real data in background
      try {
        const [dashboardRes, modelsRes, talentsRes] = await Promise.all([
          fetch('http://localhost:8084/dashboard'),
          fetch('http://localhost:8084/models/list'),
          fetch('http://localhost:8084/mcp/talents')
        ])

        if (dashboardRes.ok) {
          const dashboardData = await dashboardRes.json()
          setDashboard(dashboardData)
        }

        if (modelsRes.ok) {
          const modelsData = await modelsRes.json()
          setModels(modelsData.models || modelsData)
        }

        if (talentsRes.ok) {
          const talentsData = await talentsRes.json()
          setTalents(talentsData.talents || talentsData)
        }
      } catch (fetchError) {
        console.log('Backend not available, using default data')
      }
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files || files.length === 0) return

    // Take only the first file (individual upload)
    const file = files[0]
    console.log(`Uploading file: ${file.name}`)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://localhost:8084/dataset/ingest-single', {
        method: 'POST',
        body: formData
      })
      const result = await response.json()
      console.log('Upload result:', result)
      
      // Add to accumulated files
      setUploadedFiles(prev => [...prev, {
        ...result,
        id: `file-${Date.now()}-${Math.random()}` // String ID
      }])
    } catch (error) {
      console.error('Failed to upload file:', error)
    }

    // Clear the input so the same file can be uploaded again if needed
    event.target.value = ''
  }

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId))
  }

  const searchHuggingFaceDatasets = async (query: string) => {
    if (!query.trim()) return
    
    setHfSearchLoading(true)
    try {
      const response = await fetch(`http://localhost:8084/datasets/huggingface/search?q=${encodeURIComponent(query)}&limit=10`)
      const result = await response.json()
      setHfSearchResults(result.datasets || [])
    } catch (error) {
      console.error('Failed to search Hugging Face datasets:', error)
      setHfSearchResults([])
    } finally {
      setHfSearchLoading(false)
    }
  }

  const downloadHuggingFaceDataset = async (datasetId: string, datasetName: string) => {
    try {
      const response = await fetch('http://localhost:8084/datasets/huggingface/download', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dataset_id: datasetId,
          sample_size: 1000
        })
      })
      const result = await response.json()
      
      // Add to accumulated files
      setUploadedFiles(prev => [...prev, {
        ...result,
        id: `hf-${Date.now()}-${Math.random()}`,
        source: 'huggingface'
      }])
      
      // Track that this dataset has been added
      setAddedDatasets(prev => new Set([...prev, datasetId]))
    } catch (error) {
      console.error('Failed to download Hugging Face dataset:', error)
    }
  }

  const addExistingDataset = (dataset: any) => {
    setUploadedFiles(prev => [...prev, {
      id: `existing-${dataset.id}-${Date.now()}`,
      dataset_id: dataset.id,
      filename: dataset.name,
      rows: dataset.rows,
      has_pii: dataset.pii_masked,
      status: 'existing',
      source: 'existing',
      created_at: dataset.created_at
    }])
  }

  const startTraining = async () => {
    if (!selectedModel || uploadedFiles.length === 0) return

    const trainingParams = {
      base_model: selectedModel.id,
      dataset_ids: uploadedFiles.map(f => f.dataset_id),
      params: {
        outcome_preference: outcomePreference,
        epochs: outcomePreference === 'speed' ? 3 : outcomePreference === 'balanced' ? 5 : 10,
        learning_rate: outcomePreference === 'speed' ? 0.001 : outcomePreference === 'balanced' ? 0.0005 : 0.0001
      },
      outcome_preference: outcomePreference
    }

    try {
      const response = await fetch('http://localhost:8084/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainingParams)
      })
      const result = await response.json()

      if (result.status === 'started') {
        setTrainingStatus({
          progress: 0,
          status: 'starting',
          started_at: new Date().toLocaleTimeString()
        })
        setActiveTrainingId(result.train_id)
        setWizardStep(3)
        
        // Save training ID to localStorage for persistence
        localStorage.setItem('activeTrainingId', result.train_id)
        
        // Subscribe to WebSocket updates instead of polling
        setTimeout(() => {
          subscribeToTraining(result.train_id)
        }, 1000) // Wait for WebSocket to be ready
      }
    } catch (error) {
      console.error('Failed to start training:', error)
    }
  }


  const runEvaluation = async (talentId: string) => {
    try {
      const response = await fetch('http://localhost:8084/evaluate/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ talent_id: talentId })
      })
      setEvaluationResults(await response.json())
    } catch (error) {
      console.error('Failed to run evaluation:', error)
    }
  }

  const publishTalent = async () => {
    if (!talentName) return

    const talent = {
      id: 'talent_' + Date.now(),
      name: talentName,
      category: talentCategory,
      model_path: `/opt/talent-factory/models/${talentName.toLowerCase().replace(/\s+/g, '_')}`,
      version: '1.0.0',
      metrics: evaluationResults?.metrics || {},
      status: 'active'
    }

    try {
      const response = await fetch('http://localhost:8084/talents/publish', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(talent)
      })

      if (response.ok) {
        await loadData()
        setCurrentView('catalogue')
        resetWizard()
      }
    } catch (error) {
      console.error('Failed to publish talent:', error)
    }
  }

  const testTalent = async (talentId: string) => {
    try {
      const response = await fetch(`http://localhost:8084/mcp/talents/${talentId}/test`)
      const result = await response.json()
      
      if (response.ok) {
        alert(`Test Result:\n\nInput: ${result.test_input}\nOutput: ${result.test_output}\nConfidence: ${(result.confidence * 100).toFixed(1)}%\nResponse Time: ${result.response_time_ms}ms`)
      } else {
        alert(`Test failed: ${result.detail}`)
      }
    } catch (error) {
      console.error('Failed to test talent:', error)
      alert('Failed to test talent. Please try again.')
    }
  }


  const stopTraining = async () => {
    if (!activeTrainingId) return
    
    try {
      const response = await fetch(`http://localhost:8084/train/stop/${activeTrainingId}`, {
        method: 'POST'
      })
      
      if (response.ok) {
        const result = await response.json()
        console.log('Training stopped:', result)
        
        // Update status immediately
        setTrainingStatus({
          ...trainingStatus,
          status: 'stopped',
          stopped_at: result.stopped_at
        })
        
        // Unsubscribe from WebSocket updates
        unsubscribeFromTraining(activeTrainingId)
        
        // Clear from localStorage and state
        localStorage.removeItem('activeTrainingId')
        setActiveTrainingId(null)
        
        alert('Training stopped successfully')
      } else {
        alert('Failed to stop training. Please try again.')
      }
    } catch (error) {
      console.error('Failed to stop training:', error)
      alert('Failed to stop training. Please try again.')
    }
  }

  const resetWizard = () => {
    setWizardStep(1)
    setSelectedModel(null)
    setUploadedFiles([])
    setOutcomePreference('balanced')
    setTrainingStatus(null)
    setEvaluationResults(null)
    setTalentName('')
    setTalentCategory('general')
    setAddedDatasets(new Set())
    setHfSearchResults([])
    setHfSearchQuery('')
    setActiveTrainingId(null)
    localStorage.removeItem('activeTrainingId')
  }

  // Temporarily disable loading screen to see MCP Designer styling
  // if (loading) {
  //   return (
  //     <div className="min-h-screen bg-background flex items-center justify-center">
  //       <div className="text-center">
  //         <Brain className="h-12 w-12 text-primary mx-auto mb-4 animate-pulse" />
  //         <p className="text-muted-foreground">Loading Talent Factory...</p>
  //       </div>
  //     </div>
  //   )
  // }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-card shadow-lg border border-border rounded-2xl mx-5 mt-5 mb-8">
        <div className="max-w-7xl mx-auto px-5 py-5">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-primary mr-3" />
                     <div>
                       <h1 className="text-2xl font-bold text-primary gapi-heading">Talent Factory</h1>
                       <p className="text-sm text-muted-foreground gapi-body">Your local AI workshop</p>
                     </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-muted-foreground gapi-body">
                <span className={dashboard?.environment?.ready ? 'text-green-600' : 'text-orange-500'}>
                  {dashboard?.environment?.ready ? 'GPU Ready' : 'CPU Only'}
                </span>
                <span className="ml-3">
                  <span className={websocket ? 'text-green-600' : 'text-red-500'}>
                    {websocket ? '●' : '○'}
                  </span>
                  <span className="ml-1">
                    {websocket ? 'Live Updates' : 'Offline'}
                  </span>
                </span>
              </div>
              <Button variant="ghost" size="icon">
                <Settings className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-5 py-8">
        <Tabs value={currentView} onValueChange={(value) => setCurrentView(value as any)}>
          <TabsList className="grid w-full grid-cols-4 mb-8 bg-card border border-border rounded-xl p-1">
            <TabsTrigger value="dashboard" className="gapi-body font-semibold">Dashboard</TabsTrigger>
            <TabsTrigger value="wizard" className="gapi-body font-semibold">New Talent</TabsTrigger>
            <TabsTrigger value="catalogue" className="gapi-body font-semibold">Catalogue</TabsTrigger>
            <TabsTrigger value="settings" className="gapi-body font-semibold">Settings</TabsTrigger>
          </TabsList>

          {/* Dashboard */}
          <TabsContent value="dashboard" className="space-y-8">
            {/* Active Training Banner */}
            {activeTrainingId && (
              <div className="panel border-orange-200 bg-orange-50">
                <div className="panel-header">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></div>
                        <h3 className="text-orange-800">Training in Progress</h3>
                      </div>
                      <div className="text-sm text-orange-700 gapi-body">
                        {trainingStatus?.status === 'running' ? 
                          `Progress: ${trainingStatus.progress || 0}%` : 
                          'Preparing training...'
                        }
                      </div>
                    </div>
                    <div className="flex space-x-2">
                      <Button 
                        onClick={() => setCurrentView('wizard')}
                        variant="outline"
                        size="sm"
                        className="border-orange-300 text-orange-700 hover:bg-orange-100"
                      >
                        View Progress
                      </Button>
                      <Button 
                        onClick={stopTraining}
                        variant="outline"
                        size="sm"
                        className="border-red-300 text-red-700 hover:bg-red-100"
                      >
                        <Square className="mr-1 h-3 w-3" />
                        Stop
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="panel">
              <div className="panel-header">
                <h3>Dashboard</h3>
                <div className="text-sm text-muted-foreground gapi-body">
                  Welcome to your local AI workshop. Monitor your talents and training progress.
                </div>
              </div>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="panel">
                <div className="panel-header">
                  <h3 className="text-sm font-medium gapi-body">Active Talents</h3>
                  <Brain className="h-4 w-4 text-primary" />
                </div>
                <div className="text-2xl font-bold gapi-heading">{dashboard?.talent_count || 0}</div>
              </div>

              <div className="panel">
                <div className="panel-header">
                  <h3 className="text-sm font-medium gapi-body">Datasets</h3>
                  <Database className="h-4 w-4 text-blue-500" />
                </div>
                <div className="text-2xl font-bold gapi-heading">{dashboard?.dataset_count || 0}</div>
              </div>

              <div className="panel">
                <div className="panel-header">
                  <h3 className="text-sm font-medium gapi-body">Training Runs</h3>
                  <Zap className="h-4 w-4 text-green-500" />
                </div>
                <div className="text-2xl font-bold gapi-heading">{dashboard?.active_training || 0}</div>
              </div>

              <div className="panel">
                <div className="panel-header">
                  <h3 className="text-sm font-medium gapi-body">GPU Status</h3>
                  <Cpu className="h-4 w-4 text-purple-500" />
                </div>
                <div className={`text-lg font-bold gapi-heading ${dashboard?.environment?.ready ? 'text-green-600' : 'text-red-600'}`}>
                  {dashboard?.environment?.ready ? 'Ready' : 'Not Ready'}
                </div>
              </div>
            </div>

            {/* Environment Status */}
            <div className="panel">
              <div className="panel-header">
                <h3>Environment Status</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground gapi-body">GPU</p>
                  <p className="font-medium gapi-body">{dashboard?.environment?.gpu_name || 'None detected'}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground gapi-body">VRAM</p>
                  <p className="font-medium gapi-body">{dashboard?.environment?.vram_gb ? `${dashboard.environment.vram_gb} GB` : 'N/A'}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground gapi-body">RAM</p>
                  <p className="font-medium gapi-body">{Math.round(dashboard?.environment?.ram_gb || 0)} GB</p>
                </div>
              </div>
            </div>

            {/* Recent Training Runs */}
            <div className="panel">
              <div className="panel-header">
                <h3>Recent Training Runs</h3>
              </div>
              {dashboard?.recent_runs?.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground gapi-body">
                  <p>No training runs yet. Start by creating a new talent!</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {dashboard?.recent_runs?.map((run) => (
                    <div key={run.id} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                      <div>
                        <p className="font-medium gapi-body">{run.base_model}</p>
                        <p className="text-sm text-muted-foreground gapi-body">{run.created_at}</p>
                      </div>
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                        run.status === 'completed' ? 'bg-green-100 text-green-800' :
                        run.status === 'running' ? 'bg-blue-100 text-blue-800' :
                        run.status === 'failed' ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'
                      }`}>
                        {run.status}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </TabsContent>

          {/* New Talent Wizard */}
          <TabsContent value="wizard" className="space-y-8">
            <div>
              <h2 className="text-3xl font-bold font-heading text-foreground mb-2">New Talent Wizard</h2>
              <p className="text-muted-foreground">Create a fine-tuned model step by step.</p>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Training Wizard</CardTitle>
                <CardDescription>
                  Step {wizardStep} of 4: {
                    wizardStep === 1 ? 'Choose Model' :
                    wizardStep === 2 ? 'Prepare Data' :
                    wizardStep === 3 ? 'Train & Evaluate' : 'Publish'
                  }
                </CardDescription>
              </CardHeader>
              <CardContent>
                {/* Step 1: Choose Model */}
                {wizardStep === 1 && (
                  <div className="space-y-6">
                    <h3 className="text-lg font-semibold">Choose Base Model</h3>
                    <p className="text-muted-foreground">Select a compatible model for fine-tuning based on your hardware.</p>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {models.map((model) => (
                        <Card 
                          key={model.id} 
                          className={`cursor-pointer transition-colors ${
                            selectedModel?.id === model.id ? 'border-primary bg-primary/5' : 'hover:border-primary/50'
                          }`}
                          onClick={() => setSelectedModel(model)}
                        >
                          <CardHeader>
                            <div className="flex items-start justify-between">
                              <div>
                                <CardTitle className="text-base">{model.name}</CardTitle>
                                <CardDescription className="mt-1">{model.description}</CardDescription>
                                <div className="flex items-center mt-2 space-x-4 text-xs text-muted-foreground">
                                  <span>{model.size_gb} GB</span>
                                  <span>{model.min_vram_gb} GB VRAM min</span>
                                </div>
                              </div>
                              <div className={`w-4 h-4 rounded-full border-2 ${
                                selectedModel?.id === model.id ? 'border-primary bg-primary' : 'border-muted-foreground'
                              }`} />
                            </div>
                          </CardHeader>
                        </Card>
                      ))}
                    </div>
                    
                    <div className="flex justify-end">
                      <Button 
                        onClick={() => setWizardStep(2)} 
                        disabled={!selectedModel}
                      >
                        Next: Prepare Data
                      </Button>
                    </div>
                  </div>
                )}

                {/* Step 2: Prepare Data */}
                {wizardStep === 2 && (
                  <div className="space-y-6">
                    <h3 className="text-lg font-semibold">Prepare Training Data</h3>
                    <p className="text-muted-foreground">Upload your dataset and configure data preparation settings.</p>
                    
                    {/* Hugging Face Dataset Search */}
                    <div className="space-y-4">
                      <div className="flex items-center space-x-2">
                        <h4 className="text-lg font-semibold">Search Hugging Face Datasets</h4>
                        <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">518K+ datasets</span>
                      </div>
                      
                      <div className="flex space-x-2">
                        <input
                          type="text"
                          value={hfSearchQuery}
                          onChange={(e) => setHfSearchQuery(e.target.value)}
                          placeholder="Search for datasets (e.g., 'medical', 'code', 'conversation')"
                          className="flex-1 px-3 py-2 border border-input rounded-md focus:outline-none focus:ring-2 focus:ring-ring"
                          onKeyPress={(e) => e.key === 'Enter' && searchHuggingFaceDatasets(hfSearchQuery)}
                        />
                        <Button 
                          onClick={() => searchHuggingFaceDatasets(hfSearchQuery)}
                          disabled={!hfSearchQuery.trim() || hfSearchLoading}
                        >
                          {hfSearchLoading ? 'Searching...' : 'Search'}
                        </Button>
        </div>
                      
                      {/* Search Results */}
                      {hfSearchResults.length > 0 && (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-sm text-muted-foreground">
                            <span>Found {hfSearchResults.length} datasets</span>
                            {addedDatasets.size > 0 && (
                              <span className="text-green-600">✓ {addedDatasets.size} added</span>
                            )}
                          </div>
                          <div className="max-h-64 overflow-y-auto space-y-2">
                          {hfSearchResults.map((dataset: any, index: number) => (
                            <div key={`hf-${dataset.name}-${dataset.author}-${index}`} className="p-3 border border-border rounded-lg hover:bg-muted/50">
                              <div className="flex items-start justify-between">
                                <div className="flex-1">
                                  <div className="flex items-center space-x-2">
                                    <h5 className="font-medium text-sm">{dataset.name}</h5>
                                    <span className="text-xs text-muted-foreground">by {dataset.author}</span>
                                  </div>
                                  <p className="text-xs text-muted-foreground mt-1">{dataset.description}</p>
                                  <div className="flex items-center space-x-4 mt-2 text-xs text-muted-foreground">
                                    <span>📥 {dataset.downloads?.toLocaleString() || 0} downloads</span>
                                    <span>❤️ {dataset.likes || 0} likes</span>
                                    <span>⭐ {dataset.relevance_score?.toFixed(1) || 0} relevance</span>
                                  </div>
                                  {dataset.tags && dataset.tags.length > 0 && (
                                    <div className="flex flex-wrap gap-1 mt-2">
                                      {dataset.tags.slice(0, 3).map((tag: string, tagIndex: number) => (
                                        <span key={tagIndex} className="px-2 py-1 text-xs bg-primary/10 text-primary rounded-full">
                                          {tag}
                                        </span>
                                      ))}
                                    </div>
                                  )}
                                </div>
                                <div className="flex flex-col space-y-1 ml-4">
                                  {addedDatasets.has(dataset.id) ? (
                                    <div className="flex items-center text-xs text-green-600 bg-green-50 px-2 py-1 rounded">
                                      <span className="mr-1">✓</span>
                                      Added
                                    </div>
                                  ) : (
                                    <Button
                                      size="sm"
                                      onClick={() => downloadHuggingFaceDataset(dataset.id, dataset.name)}
                                      className="text-xs"
                                    >
                                      <Download className="h-3 w-3 mr-1" />
                                      Add
                                    </Button>
                                  )}
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => window.open(dataset.url, '_blank')}
                                    className="text-xs"
                                  >
                                    View
                                  </Button>
                                </div>
                              </div>
                            </div>
                          ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Existing Datasets */}
                    {existingDatasets.length > 0 && (
                      <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                          <h4 className="text-lg font-semibold">Use Existing Datasets</h4>
                          <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full">{existingDatasets.length} available</span>
                        </div>
                        <p className="text-sm text-muted-foreground">Reuse previously uploaded datasets for training</p>
                        
                        <div className="max-h-60 overflow-y-auto border rounded-lg">
                          <div className="divide-y">
                            {existingDatasets.slice(0, 10).map((dataset) => (
                              <div key={dataset.id} className="p-4 hover:bg-muted/50">
                                <div className="flex items-center justify-between">
                                  <div className="flex-1">
                                    <div className="flex items-center space-x-2">
                                      <h5 className="font-medium text-sm">{dataset.name}</h5>
                                      {dataset.pii_masked && (
                                        <span className="text-xs bg-orange-100 text-orange-800 px-2 py-1 rounded-full">PII Masked</span>
                                      )}
                                    </div>
                                    <p className="text-xs text-muted-foreground mt-1">
                                      {dataset.rows} examples • {dataset.source} • {new Date(dataset.created_at).toLocaleDateString()}
                                    </p>
                                  </div>
                                  <Button
                                    size="sm"
                                    onClick={() => addExistingDataset(dataset)}
                                    className="text-xs"
                                  >
                                    <Plus className="h-3 w-3 mr-1" />
                                    Add
                                  </Button>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                        {existingDatasets.length > 10 && (
                          <p className="text-xs text-muted-foreground text-center">
                            Showing first 10 of {existingDatasets.length} datasets
                          </p>
                        )}
                      </div>
                    )}

                    <div className="relative">
                      <div className="absolute inset-0 flex items-center">
                        <span className="w-full border-t" />
                      </div>
                      <div className="relative flex justify-center text-xs uppercase">
                        <span className="bg-background px-2 text-muted-foreground">Or upload your own files</span>
                      </div>
                    </div>

                    {/* File Upload */}
                    <div className="border-2 border-dashed border-muted-foreground rounded-lg p-8 text-center">
                      <input 
                        type="file" 
                        onChange={handleFileUpload} 
                        className="hidden" 
                        id="dataset-upload" 
                        accept=".json,.csv,.txt,.md,.pdf,.docx,.rtf,.html,.xml" 
                      />
                      <label htmlFor="dataset-upload" className="cursor-pointer">
                        <Upload className="mx-auto h-12 w-12 text-muted-foreground" />
                        <p className="mt-2 text-sm text-muted-foreground">
                          <span className="font-medium text-primary hover:text-primary/80">Click to upload</span> or drag and drop
                        </p>
                        <p className="text-xs text-muted-foreground">Supported formats: JSON, CSV, TXT, MD, PDF, DOCX, RTF, HTML, XML</p>
                        <p className="text-xs text-muted-foreground">Talent Factory will convert unstructured data into trainable format</p>
                        <p className="text-xs text-muted-foreground mt-2 font-medium">Upload one file at a time from different folders</p>
                      </label>
                    </div>

                    {/* Uploaded Files List */}
                    {uploadedFiles.length > 0 && (
                      <Card>
                        <CardContent className="pt-6">
                          <div className="space-y-4">
                            <div className="flex items-center justify-between">
                              <h4 className="font-medium">Uploaded Files ({uploadedFiles.length})</h4>
                              <p className="text-sm text-muted-foreground">
                                Total: {uploadedFiles.reduce((sum, f) => sum + (f.rows || 0), 0)} training examples
                              </p>
                            </div>
                            
                            <div className="space-y-2">
                              {uploadedFiles.map((file: any, index: number) => (
                                <div key={file.id || `file-${index}`} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                                  <div className="flex-1">
                                    <div className="flex items-center space-x-2">
                                      <p className="font-medium text-sm">{file.filename}</p>
                                      {file.source === 'huggingface' && (
                                        <span className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded-full">HF Dataset</span>
                                      )}
                                    </div>
                                    <p className="text-xs text-muted-foreground">
                                      {file.rows} training examples • {file.mime_type || 'Unknown type'}
                                    </p>
                                  </div>
                                  <div className="flex items-center space-x-2">
                                    {file.has_pii ? (
                                      <span className="px-2 py-1 text-xs font-medium bg-red-100 text-red-800 rounded-full">PII Detected</span>
                                    ) : (
                                      <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full">Clean</span>
                                    )}
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      onClick={() => removeFile(file.id)}
                                      className="text-red-600 hover:text-red-800"
                                    >
                                      <X className="h-4 w-4" />
                                    </Button>
        </div>
                                </div>
                              ))}
                            </div>
                            
                            <div className="flex items-center space-x-2 pt-2 border-t">
                              <span className="text-xs text-muted-foreground">
                                Talent Factory automatically converts unstructured data into trainable format
                              </span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    {/* Outcome Preference */}
                    <div className="space-y-4">
                      <label className="block text-sm font-medium">Training Outcome Preference</label>
                      <div className="grid grid-cols-3 gap-4">
                        {(['speed', 'balanced', 'quality'] as const).map((pref) => (
                          <Card 
                            key={pref}
                            className={`cursor-pointer transition-colors ${
                              outcomePreference === pref ? 'border-primary bg-primary/5' : 'hover:border-primary/50'
                            }`}
                            onClick={() => setOutcomePreference(pref)}
                          >
                            <CardContent className="pt-6 text-center">
                              <div className="text-lg font-medium capitalize">{pref}</div>
                              <div className="text-xs text-muted-foreground mt-1">
                                {pref === 'speed' ? 'Fast training, good results' :
                                 pref === 'balanced' ? 'Good speed and quality' :
                                 'Best results, longer training'}
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    </div>
                    
                    <div className="flex justify-between">
                      <Button variant="outline" onClick={() => setWizardStep(1)}>
                        Back
                      </Button>
                      <Button 
                        onClick={() => setWizardStep(3)} 
                        disabled={uploadedFiles.length === 0}
                      >
                        Next: Train & Evaluate
                      </Button>
                    </div>
                  </div>
                )}

                {/* Step 3: Train & Evaluate */}
                {wizardStep === 3 && (
                  <div className="space-y-6">
                    <h3 className="text-lg font-semibold">Train & Evaluate</h3>
                    <p className="text-muted-foreground">Start training your model and monitor progress.</p>
                    
                    {/* Training Configuration */}
                    <Card>
                      <CardHeader>
                        <CardTitle>Training Configuration</CardTitle>
                      </CardHeader>
                      <CardContent>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="text-muted-foreground">Base Model:</span>
                                <span className="ml-2 font-medium">{selectedModel?.name}</span>
                              </div>
                              <div>
                                <span className="text-muted-foreground">Files:</span>
                                <span className="ml-2 font-medium">{uploadedFiles.length} files</span>
                              </div>
                              <div>
                                <span className="text-muted-foreground">Preference:</span>
                                <span className="ml-2 font-medium capitalize">{outcomePreference}</span>
                              </div>
                              <div>
                                <span className="text-muted-foreground">Total Examples:</span>
                                <span className="ml-2 font-medium">{uploadedFiles.reduce((sum, f) => sum + (f.rows || 0), 0)}</span>
                              </div>
                            </div>
                      </CardContent>
                    </Card>

                    {/* Training Progress */}
                    {trainingStatus && (
                      <Card>
                        <CardHeader>
                          <CardTitle className="flex items-center">
                            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse mr-2"></div>
                            Training Progress
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-4">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium">Progress</span>
                              <span className="text-sm text-muted-foreground">{trainingStatus.progress || 0}%</span>
                            </div>
                            <Progress value={trainingStatus.progress || 0} />
                            
                            <div className="grid grid-cols-2 gap-4 text-xs">
                              <div>
                                <span className="text-muted-foreground">Status:</span>
                                <div className="font-medium capitalize">{trainingStatus.status || 'Unknown'}</div>
                              </div>
                              <div>
                                <span className="text-muted-foreground">Started:</span>
                                <div className="font-medium">{trainingStatus.started_at || 'Unknown'}</div>
                              </div>
                            </div>
                            
                            {trainingStatus.status === 'running' && (
                              <div className="bg-blue-50 p-3 rounded-lg">
                                <div className="text-sm text-blue-800">
                                  <strong>Training is running!</strong> Your laptop fans are working hard. 
                                  This may take 30-60 minutes depending on your hardware.
                                </div>
                              </div>
                            )}
                            
                            {trainingStatus.status === 'preparing' && (
                              <div className="bg-yellow-50 p-3 rounded-lg">
                                <div className="text-sm text-yellow-800">
                                  <strong>Preparing training...</strong> Setting up the model and dataset.
                                </div>
                              </div>
                            )}
                            
                            {/* Stop Training Button */}
                            {(trainingStatus.status === 'running' || trainingStatus.status === 'preparing') && (
                              <div className="flex justify-center pt-4">
                                <Button 
                                  onClick={stopTraining}
                                  variant="outline"
                                  className="border-red-300 text-red-700 hover:bg-red-50"
                                >
                                  <Square className="mr-2 h-4 w-4" />
                                  Stop Training
                                </Button>
                              </div>
                            )}
                            
                            {trainingStatus.status === 'stopped' && (
                              <div className="bg-red-50 p-3 rounded-lg">
                                <div className="text-sm text-red-800">
                                  <strong>Training stopped</strong> by user at {trainingStatus.stopped_at}
                                </div>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    {/* Start Training Button */}
                    {!trainingStatus && (
                      <div className="text-center">
                        <Button onClick={startTraining} size="lg">
                          <Play className="mr-2 h-4 w-4" />
                          Start Training
                        </Button>
                      </div>
                    )}

                    {/* Evaluation Results */}
                    {evaluationResults && (
                      <Card>
                        <CardHeader>
                          <CardTitle className="flex items-center">
                            <CheckCircle className="mr-2 h-5 w-5 text-green-500" />
                            Evaluation Results
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <span className="text-muted-foreground">Safety Score:</span>
                              <span className="ml-2 font-medium text-green-600">
                                {(evaluationResults.safety_score * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div>
                              <span className="text-muted-foreground">Rubric Passed:</span>
                              <span className="ml-2 font-medium text-green-600">
                                {evaluationResults.rubric_passed ? 'Yes' : 'No'}
                              </span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    
                    <div className="flex justify-between">
                      <Button variant="outline" onClick={() => setWizardStep(2)}>
                        Back
                      </Button>
                      <Button 
                        onClick={() => setWizardStep(4)} 
                        disabled={!evaluationResults}
                      >
                        Next: Publish
                      </Button>
                    </div>
                  </div>
                )}

                {/* Step 4: Publish */}
                {wizardStep === 4 && (
                  <div className="space-y-6">
                    <h3 className="text-lg font-semibold">Publish Talent</h3>
                    <p className="text-muted-foreground">Publish your trained model to the Talent Catalogue.</p>
                    
                    {/* Talent Details */}
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium mb-2">Talent Name</label>
                        <input 
                          type="text" 
                          value={talentName}
                          onChange={(e) => setTalentName(e.target.value)}
                          className="w-full px-3 py-2 border border-input rounded-md focus:outline-none focus:ring-2 focus:ring-ring"
                          placeholder="Enter a name for your talent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Category</label>
                        <select 
                          value={talentCategory}
                          onChange={(e) => setTalentCategory(e.target.value)}
                          className="w-full px-3 py-2 border border-input rounded-md focus:outline-none focus:ring-2 focus:ring-ring"
                        >
                          <option value="general">General</option>
                          <option value="coding">Coding</option>
                          <option value="writing">Writing</option>
                          <option value="analysis">Analysis</option>
                          <option value="creative">Creative</option>
                        </select>
                      </div>
                    </div>

                    {/* Publish Summary */}
                    <Card>
                      <CardHeader>
                        <CardTitle>Publish Summary</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Name:</span>
                            <span className="font-medium">{talentName || 'Untitled Talent'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Category:</span>
                            <span className="font-medium capitalize">{talentCategory}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Base Model:</span>
                            <span className="font-medium">{selectedModel?.name}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Safety Score:</span>
                            <span className="font-medium">
                              {(evaluationResults?.safety_score * 100).toFixed(1)}%
                            </span>
                          </div>
        </div>
                      </CardContent>
                    </Card>
                    
                    <div className="flex justify-between">
                      <Button variant="outline" onClick={() => setWizardStep(3)}>
                        Back
                      </Button>
                      <Button 
                        onClick={publishTalent} 
                        disabled={!talentName}
                      >
                        Publish Talent
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Talent Catalogue */}
          <TabsContent value="catalogue" className="space-y-8">
            <div>
              <h2 className="text-3xl font-bold font-heading text-foreground mb-2">Talent Catalogue</h2>
              <p className="text-muted-foreground">Browse and manage your published talents.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {talents.map((talent) => (
                <Card key={talent.id}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-lg">{talent.name}</CardTitle>
                        <CardDescription className="capitalize">{talent.category}</CardDescription>
                      </div>
                      <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full">
                        {talent.status}
                      </span>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 text-sm text-muted-foreground mb-4">
                      <div className="flex justify-between">
                        <span>Version:</span>
                        <span>{talent.version || '1.0.0'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Created:</span>
                        <span>{new Date(talent.created_at).toLocaleDateString()}</span>
                      </div>
                    </div>
                    
                    <div className="flex space-x-2">
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="flex-1"
                        onClick={() => testTalent(talent.id)}
                      >
                        <Play className="mr-1 h-3 w-3" />
                        Test
                      </Button>
                      <Button size="sm" className="flex-1">
                        <Download className="mr-1 h-3 w-3" />
                        Export
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {talents.length === 0 && (
              <Card>
                <CardContent className="text-center py-12">
                  <Brain className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium mb-2">No talents yet</h3>
                  <p className="text-muted-foreground mb-6">Get started by creating your first talent.</p>
                  <Button onClick={() => setCurrentView('wizard')}>
                    <Plus className="mr-2 h-4 w-4" />
                    Create Talent
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Settings */}
          <TabsContent value="settings" className="space-y-8">
            <div>
              <h2 className="text-3xl font-bold font-heading text-foreground mb-2">Settings</h2>
              <p className="text-muted-foreground">Configure Talent Factory settings and preferences.</p>
            </div>

            <div className="space-y-6">
              {/* Network & Access */}
              <Card>
                <CardHeader>
                  <CardTitle>Network & Access</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Hostname</label>
                    <input 
                      type="text" 
                      value="talentfactory.local" 
                      className="w-full px-3 py-2 border border-input rounded-md bg-muted"
                      readOnly
                    />
                    <p className="text-xs text-muted-foreground mt-1">Advertised via mDNS</p>
    </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Access Control</label>
                    <select className="w-full px-3 py-2 border border-input rounded-md">
                      <option>LAN Only (192.168.x.x, 10.x.x.x)</option>
                      <option>Localhost Only</option>
                      <option>Custom Range</option>
                    </select>
                  </div>
                </CardContent>
              </Card>

              {/* Storage */}
              <Card>
                <CardHeader>
                  <CardTitle>Storage</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Models Directory</label>
                    <input 
                      type="text" 
                      value="/opt/talent-factory/models" 
                      className="w-full px-3 py-2 border border-input rounded-md bg-muted"
                      readOnly
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Datasets Directory</label>
                    <input 
                      type="text" 
                      value="/opt/talent-factory/datasets" 
                      className="w-full px-3 py-2 border border-input rounded-md bg-muted"
                      readOnly
                    />
                  </div>
                </CardContent>
              </Card>

              {/* Security */}
              <Card>
                <CardHeader>
                  <CardTitle>Security</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm font-medium">PII Detection</label>
                      <p className="text-xs text-muted-foreground">Automatically detect and mask PII in datasets</p>
                    </div>
                    <input type="checkbox" defaultChecked className="w-4 h-4 text-primary" />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm font-medium">Audit Logging</label>
                      <p className="text-xs text-muted-foreground">Log all actions for compliance</p>
                    </div>
                    <input type="checkbox" defaultChecked className="w-4 h-4 text-primary" />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm font-medium">Local-First</label>
                      <p className="text-xs text-muted-foreground">Keep all data local by default</p>
                    </div>
                    <input type="checkbox" defaultChecked className="w-4 h-4 text-primary" />
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}