# Video Demo Instructions
## IBM Watson AI Challenge - Financial Fraud Detection

This guide helps you record the demo for your video submission.

---

## 🎬 Two Demo Options

### Option 1: Standalone Demo (Recommended for Quick Recording)
**No API server needed - runs directly**

```bash
cd Financial-Fraud-Detector
python demo.py --video
```

**What it shows:**
- Scene 3: Normal transaction with low risk score
- Scene 4: Fraudulent transaction with high risk score
- JSON formatted output ready for screen recording
- Narration text included

**Best for:**
- Quick video recording
- No setup required
- Clean, formatted output

---

### Option 2: Live API Demo (More Impressive)
**Shows actual REST API calls**

**Step 1: Start API Server**
```bash
# Terminal 1
cd Financial-Fraud-Detector
python src/api_service.py
```

**Step 2: Run API Demo**
```bash
# Terminal 2
cd Financial-Fraud-Detector
python video_demo_api.py
```

**What it shows:**
- Real HTTP POST requests to API
- Actual response times
- Production-ready REST API
- More impressive for judges

**Best for:**
- Showing production-ready system
- Demonstrating API integration
- More technical audience

---

## 📹 Recording Steps

### 1. Prepare Your Environment

```bash
# Install dependencies
cd Financial-Fraud-Detector
pip install -r requirements.txt

# Test the demo
python demo.py --video
```

### 2. Set Up Screen Recording

**Recommended Tools:**
- **Windows:** OBS Studio (free), Camtasia
- **Mac:** QuickTime, ScreenFlow
- **Linux:** OBS Studio, SimpleScreenRecorder

**Settings:**
- Resolution: 1920x1080 (1080p)
- Frame rate: 30 fps
- Audio: Clear microphone
- Cursor: Highlight enabled

### 3. Recording Checklist

- [ ] Close unnecessary applications
- [ ] Clear terminal history
- [ ] Set terminal font size large (14-16pt)
- [ ] Use high contrast theme (dark background, light text)
- [ ] Test audio levels
- [ ] Have script ready (VIDEO_SCRIPT_GUIDE.md)

### 4. Record Scene 3 & 4

**For Standalone Demo:**
```bash
python demo.py --video
```

**For API Demo:**
```bash
# Terminal 1 (keep visible in corner)
python src/api_service.py

# Terminal 2 (main focus)
python video_demo_api.py
```

**While recording, narrate:**
- Scene 3: "Here's a normal transaction... Risk score 8... Approved in 145ms"
- Scene 4: "Now a suspicious transaction... Risk score 92... Blocked with explanations"

---

## 🎯 What to Show in Video

### Scene 3: Normal Transaction (30 seconds)

**Show on screen:**
```json
{
  "transaction_id": "TXN_001",
  "user_id": "DEMO_USER",
  "amount": 49.99,
  "merchant_category": "grocery",
  "location": "US",
  "device_id": "DEVICE_A"
}
```

**Response:**
```json
{
  "risk_score": 8,
  "decision": "approve",
  "action": "none",
  "confidence": "high",
  "latency_ms": 145
}
```

**Narrate:**
> "Risk score: 8 out of 100. Decision: Approve. No customer friction. Transaction completed in 145 milliseconds."

---

### Scene 4: Fraudulent Transaction (45 seconds)

**Show on screen:**
```json
{
  "transaction_id": "TXN_002",
  "user_id": "DEMO_USER",
  "amount": 5000.00,
  "merchant_category": "gambling",
  "location": "CN",
  "device_id": "UNKNOWN"
}
```

**Response:**
```json
{
  "risk_score": 92,
  "decision": "block",
  "action": "fraud_alert",
  "top_risk_factors": [
    {"factor": "unusual_amount", "contribution": 35},
    {"factor": "high_risk_merchant", "contribution": 28},
    {"factor": "unknown_device", "contribution": 18},
    {"factor": "foreign_location", "contribution": 11}
  ]
}
```

**Narrate:**
> "Risk score: 92. Decision: Block. The system immediately identifies this as fraud and provides explainable reasons—unusual amount, high-risk merchant, unknown device, and foreign location. The fraud team gets an instant alert with all the context they need."

---

## 💡 Pro Tips

### Terminal Setup
```bash
# Make terminal look professional
# Windows PowerShell
$Host.UI.RawUI.BackgroundColor = "Black"
$Host.UI.RawUI.ForegroundColor = "Green"
Clear-Host

# Increase font size in terminal settings (14-16pt)
```

### Recording Tips
1. **Speak Clearly**: Enunciate, don't rush
2. **Pause Between Scenes**: Give viewers time to read
3. **Highlight Important Parts**: Use cursor to point at key values
4. **Keep It Simple**: Don't over-explain, let the demo speak
5. **Practice First**: Do a dry run before final recording

### Common Issues

**Issue: Output too fast**
```python
# Add pauses in demo.py if needed
import time
time.sleep(2)  # Pause for 2 seconds
```

**Issue: Terminal text too small**
- Increase terminal font size to 14-16pt
- Use zoom if needed

**Issue: Colors not visible**
- Use high contrast theme
- Dark background with light text works best

---

## 🎬 Alternative: Using Postman

If you prefer a GUI tool:

1. **Install Postman** (free)
2. **Import these requests:**

**Normal Transaction:**
- Method: POST
- URL: `http://localhost:5000/api/v1/predict`
- Body (JSON):
```json
{
  "transaction_id": "TXN_001",
  "user_id": "DEMO_USER",
  "amount": 49.99,
  "merchant_category": "grocery",
  "location": "US",
  "device_id": "DEVICE_A"
}
```

**Fraudulent Transaction:**
- Method: POST
- URL: `http://localhost:5000/api/v1/predict`
- Body (JSON):
```json
{
  "transaction_id": "TXN_002",
  "user_id": "DEMO_USER",
  "amount": 5000.00,
  "merchant_category": "gambling",
  "location": "CN",
  "device_id": "UNKNOWN"
}
```

---

## 📋 Quick Reference

### Commands Summary

```bash
# Option 1: Standalone Demo
python demo.py --video

# Option 2: API Demo
# Terminal 1:
python src/api_service.py

# Terminal 2:
python video_demo_api.py

# Show curl commands:
python video_demo_api.py --curl
```

### Expected Output

| Transaction | Risk Score | Decision | Latency |
|-------------|------------|----------|---------|
| TXN_001 (Normal) | ~8 | Approve | ~145ms |
| TXN_002 (Fraud) | ~92 | Block | ~150ms |

---

## ✅ Pre-Recording Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Demo script tested and works
- [ ] Terminal font size increased (14-16pt)
- [ ] Screen recording software ready
- [ ] Microphone tested
- [ ] Script/narration prepared
- [ ] Quiet recording environment
- [ ] Unnecessary apps closed

---

## 🚀 Ready to Record!

1. **Run the demo**: `python demo.py --video`
2. **Start recording**: Begin screen capture
3. **Narrate clearly**: Follow the script
4. **Show the results**: Let output display fully
5. **Stop recording**: Save your video

**Your demo is ready for the IBM Watson AI Challenge! 🎉**

---

## 📞 Need Help?

**Test your setup:**
```bash
# Quick test
python demo.py --video

# Should see:
# ✅ Scene 3 output (normal transaction)
# ✅ Scene 4 output (fraudulent transaction)
# ✅ Formatted JSON responses
# ✅ Narration text
```

**If something doesn't work:**
1. Check Python version: `python --version` (need 3.10+)
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check file paths: Make sure you're in `Financial-Fraud-Detector` directory

---

**Good luck with your video recording! 🎬🚀**
