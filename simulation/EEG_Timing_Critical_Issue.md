# CRITICAL: 2.6 Second Frame Drop Impact on EEG Data Alignment

## The Problem

You're seeing: `SYNC WARNING: Frame took 2646.6ms (expected 16.7ms)`

This is **catastrophic** for EEG data alignment. Here's why:

## Why This Destroys EEG/ERP Analysis

### 1. LSL Marker Timing Uncertainty
```
What you think is happening:
T=0ms: Trial ends
T=1ms: LSL marker "TRIAL_COMPLETE" sent
T=2ms: New trial setup begins

What's actually happening:
T=0ms: Trial ends
T=1ms: LSL marker "TRIAL_COMPLETE" sent
T=1-2647ms: Game frozen, no updates
T=2647ms: New trial actually starts
```

Your LSL markers are **2.6 seconds off** from visual events!

### 2. EEG Epoch Extraction Failure
When you extract epochs in your analysis:
```python
# You extract -200ms to 800ms around "CUE_START"
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8)
```

But the actual cue might have appeared 2.6 seconds later than the marker!

### 3. ERP Component Misalignment
- ErrP peaks at ~250-350ms after error
- But if the error display is delayed by 2600ms...
- You're looking at the wrong time window entirely
- You might be analyzing alpha waves instead of ErrPs!

## Where The Delay Happens

Based on the code, the delay occurs:

1. **Trial completion animation** (~2.6s total)
2. **Object creation between trials** (minor)
3. **Surface creation for ITI** (minor)

## Impact on Your Research

This makes your data **unpublishable** because:

1. **Reviewers will reject** - 2.6s timing uncertainty is unacceptable
2. **Cannot replicate** - Other labs won't see the same ERPs
3. **Wrong components** - You're not measuring what you think
4. **Statistical noise** - Averaging misaligned data destroys signal

## Critical Test

Add this to verify the issue:

```python
# In show_trial_complete, at the start:
complete_start_real = time.perf_counter()
complete_start_lsl = local_clock()

# At the end:
complete_end_real = time.perf_counter()
complete_end_lsl = local_clock()

print(f"Animation took: {(complete_end_real - complete_start_real)*1000:.1f}ms")
print(f"LSL clock drift: {(complete_end_lsl - complete_start_lsl)*1000:.1f}ms")
```

## Solutions

### Option 1: Remove Blocking Animation
Replace the 2.6s animation with instant feedback

### Option 2: Send Markers AFTER Delays
```python
# Wrong:
push_immediate("TRIAL_COMPLETE")
show_trial_complete()  # 2.6s delay

# Right:
show_trial_complete()  # 2.6s delay
push_immediate("TRIAL_COMPLETE")  # Now aligned!
```

### Option 3: Use Threading
Run animations in separate thread while game continues

## The Bottom Line

**This MUST be fixed before collecting any EEG data!**

A 2.6 second uncertainty means:
- Your "250ms ErrP" could appear anywhere from -2350ms to +2850ms
- That's a 5.2 second window of uncertainty
- No statistical analysis can fix this
- Your data is essentially random noise

## Verification After Fix

After fixing, verify with:
```python
# All frame times should be <20ms
# No frame should exceed 50ms for EEG quality
```

Remember: For publishable EEG research, you need <2ms precision. Currently you have 2600ms imprecision - that's 1300x worse than acceptable!