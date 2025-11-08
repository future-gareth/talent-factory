# WebSocket Streaming Fix

## The Problem

The user correctly identified that the UI was **polling** for training status updates every 5 seconds, instead of properly using **WebSocket streaming** for real-time updates.

### Root Cause

The backend had full WebSocket support with broadcasting capabilities, but the frontend was **NOT subscribing** to training updates. Here's what was happening:

1. ✅ Backend broadcasts training progress via WebSocket
2. ✅ Frontend connects to WebSocket endpoint
3. ❌ **Frontend never subscribes to specific training updates**
4. ❌ Backend sees no subscribers, so doesn't send updates
5. ❌ Frontend falls back to polling every 5 seconds

## The Solution

### Backend WebSocket Architecture

The backend requires clients to explicitly **subscribe** to training updates by sending:

```json
{
  "type": "subscribe_training",
  "train_id": "..."
}
```

Once subscribed, the backend broadcasts real-time updates:

```json
{
  "type": "training_update",
  "train_id": "...",
  "data": {
    "progress": 45,
    "status": "running",
    "details": {...}
  }
}
```

### Frontend Changes

#### 1. Added Subscription Functions

```typescript
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
```

#### 2. Subscribe on WebSocket Connect

When the WebSocket connects (or reconnects), automatically subscribe to any active training:

```typescript
ws.onopen = () => {
  console.log('WebSocket connected')
  setWebsocket(ws)
  
  // Subscribe to active training if we have one
  const storedTrainingId = localStorage.getItem('activeTrainingId')
  if (storedTrainingId) {
    ws.send(JSON.stringify({
      type: 'subscribe_training',
      train_id: storedTrainingId
    }))
  }
}
```

#### 3. Subscribe When Training Starts

When a new training is started, immediately subscribe to its updates:

```typescript
if (result.status === 'started') {
  // ... set state ...
  
  // Subscribe to WebSocket updates for real-time progress
  const trySubscribe = () => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      subscribeToTraining(result.train_id)
    } else {
      // Retry if WebSocket isn't ready yet
      setTimeout(trySubscribe, 500)
    }
  }
  trySubscribe()
}
```

#### 4. Handle Subscription Confirmations

Added handling for subscription confirmation messages:

```typescript
case 'subscription_confirmed':
  console.log('Subscription confirmed for:', message.train_id, message.status)
  break
```

#### 5. Reduced Polling Frequency

Since WebSocket now provides real-time updates, polling is reduced from **5 seconds** to **30 seconds** as a fallback safety net:

```typescript
// Poll every 30 seconds as fallback (WebSocket provides real-time updates)
// This is just a safety net in case WebSocket disconnects or misses updates
const pollInterval = setInterval(pollTrainingStatus, 30000)
```

## Benefits

### Before (Polling)
- ❌ 5-second delay between updates
- ❌ 12 unnecessary HTTP requests per minute
- ❌ Higher server load
- ❌ Slower feedback for users
- ❌ Potential missed updates between polls

### After (WebSocket Streaming)
- ✅ **Real-time updates** (< 1 second latency)
- ✅ **Minimal server load** (only sends when data changes)
- ✅ **Instant feedback** for users
- ✅ **Efficient bandwidth usage**
- ✅ **Graceful fallback** with 30-second polling safety net

## Testing

To verify the fix is working:

1. Open the Talent Factory UI at `http://localhost:3200`
2. Open browser DevTools → Console
3. Start a training
4. Look for these console messages:

```
WebSocket connected
Subscribing to training: <train_id>
Subscription confirmed for: <train_id> subscribed
WebSocket message received: {type: "training_update", ...}
Training update: running 35%
Training update: running 45%
...
```

You should see training updates streaming in real-time via WebSocket, NOT via polling requests.

## Files Modified

- `tools/talent-factory/ui/src/app/page.tsx`:
  - Added `subscribeToTraining()` and `unsubscribeFromTraining()` functions
  - Modified `initializeWebSocket()` to subscribe on connection
  - Updated `startTraining()` to subscribe when training starts
  - Added subscription confirmation handling in `handleWebSocketMessage()`
  - Reduced polling interval from 5s to 30s

## Summary

The system now properly uses **WebSocket streaming** for real-time training progress updates, with polling retained only as a safety fallback. This provides a much better user experience with instant feedback and significantly reduced server load.

