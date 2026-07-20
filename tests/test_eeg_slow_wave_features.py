from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import numpy as np

from src import eeg_processing
from src.lib import eeg_features
from src.lib.swa.swa_FindSWRef import swa_FindSWRef
from src.lib.swa.swa_getInfoDefaults import swa_getInfoDefaults


class SlowWaveFeatureTests(unittest.TestCase):
    def test_summarize_slow_waves_returns_fixed_numeric_features(self):
        waves = [
            {
                'Ref_DownInd': 10,
                'Ref_UpInd': 110,
                'Ref_PeakAmp': -80,
                'Ref_P2PAmp': 130,
                'Ref_NegSlope': -4,
                'Ref_PosSlope': 5,
            },
            {
                'Ref_DownInd': 200,
                'Ref_UpInd': 350,
                'Ref_PeakAmp': -120,
                'Ref_P2PAmp': 190,
                'Ref_NegSlope': -8,
                'Ref_PosSlope': 9,
            },
        ]

        features = eeg_features.summarize_slow_waves(
            waves, fs=100, signal_duration_seconds=120
        )

        self.assertEqual(tuple(features), eeg_features.SLOW_WAVE_FEATURE_NAMES)
        self.assertEqual(features['TotalSW'], 2.0)
        self.assertEqual(features['SWdensity'], 1.0)
        self.assertEqual(features['SWpeakAmp_mean'], -100.0)
        self.assertEqual(features['SWpeakAmp_std'], 20.0)
        self.assertEqual(features['SWp2p_mean'], 160.0)
        self.assertEqual(features['SWnegSlope_mean'], -6.0)
        self.assertEqual(features['SWposSlope_mean'], 7.0)
        self.assertAlmostEqual(features['SWduration_mean'], 1.25)
        self.assertAlmostEqual(features['SWduration_std'], 0.25)

    def test_summarize_no_waves_distinguishes_absence_from_missing_morphology(self):
        features = eeg_features.summarize_slow_waves(
            [], fs=200, signal_duration_seconds=300
        )

        self.assertEqual(features['TotalSW'], 0.0)
        self.assertEqual(features['SWdensity'], 0.0)
        for name in eeg_features.SLOW_WAVE_FEATURE_NAMES[2:]:
            self.assertTrue(np.isnan(features[name]), name)

    def test_swa_reference_detector_uses_channels_by_samples_orientation(self):
        fs = 100
        seconds = 20
        time = np.arange(fs * seconds) / fs
        reference = 100.0 * np.sin(2 * np.pi * time)

        info = swa_getInfoDefaults({}, 'SW', method='envelope')
        info['Recording'] = {'sRate': fs}
        info['Parameters']['Ref_InspectionPoint'] = 'ZC'
        info['Parameters']['Ref_AmplitudeCriteria'] = 'absolute'
        info['Parameters']['Ref_AmplitudeAbsolute'] = 50.0
        data = {'SWRef': reference[np.newaxis, :]}

        with redirect_stdout(io.StringIO()):
            _, _, waves = swa_FindSWRef(data, info)

        self.assertGreater(len(waves), 0)
        self.assertTrue(all(0 <= wave['Ref_PeakInd'] < reference.size for wave in waves))

    def test_get_sw_features_detects_a_synthetic_slow_wave_signal(self):
        fs = 100
        time = np.arange(fs * 20) / fs
        signal = 100.0 * np.sin(2 * np.pi * time)

        features = eeg_features.get_SW_features(signal, fs)

        self.assertGreater(features['TotalSW'], 10)
        self.assertGreater(features['SWdensity'], 30)
        self.assertLess(features['SWpeakAmp_mean'], -80)
        self.assertGreater(features['SWp2p_mean'], 160)
        self.assertTrue(0.4 < features['SWduration_mean'] < 0.6)

    def test_get_sw_features_aggregates_swa_events(self):
        fs = 100
        signal = np.zeros(fs * 60, dtype=float)
        waves = [{
            'Ref_DownInd': 100,
            'Ref_UpInd': 180,
            'Ref_PeakInd': 140,
            'Ref_PeakAmp': -75,
            'Ref_P2PAmp': 125,
            'Ref_NegSlope': -3,
            'Ref_PosSlope': 4,
        }]

        with (
            patch.object(
                eeg_features.swa_CalculateReference,
                'swa_CalculateReference',
                return_value=(signal[np.newaxis, :], {
                    'Recording': {'sRate': fs},
                    'Parameters': {},
                    'Electrodes': ['EEG'],
                }),
            ),
            patch.object(
                eeg_features.swa_FindSWRef,
                'swa_FindSWRef',
                side_effect=lambda data, info: (data, info, waves),
            ),
            patch.object(
                eeg_features.swa_FindSWChannels,
                'swa_FindSWChannels',
                side_effect=lambda data, info, detected, flag_progress: (
                    data, info, detected
                ),
            ),
        ):
            features = eeg_features.get_SW_features(signal, fs)

        self.assertEqual(tuple(features), eeg_features.SLOW_WAVE_FEATURE_NAMES)
        self.assertEqual(features['TotalSW'], 1.0)
        self.assertEqual(features['SWdensity'], 1.0)
        self.assertEqual(features['SWpeakAmp_mean'], -75.0)

    def test_sw_failure_does_not_discard_other_channel_metrics(self):
        fs = 200
        time = np.arange(fs * 30) / fs
        signal = (
            20.0 * np.sin(2 * np.pi * 10 * time)
            + 5.0 * np.sin(2 * np.pi * time)
        )

        with patch.object(
            eeg_features,
            'get_SW_features',
            side_effect=RuntimeError('detector failed'),
        ):
            metrics = eeg_processing._extract_channel_metrics(signal, fs)

        self.assertIsNotNone(metrics)
        self.assertTrue(np.isfinite(metrics['Relative_Delta_Power']))
        for name in eeg_features.SLOW_WAVE_FEATURE_NAMES:
            self.assertTrue(np.isnan(metrics[name]), name)

    def test_all_slow_wave_features_are_exposed_for_each_eeg_channel(self):
        for channel_name in eeg_processing.EEG_CHANNEL_SPECS:
            for feature_name in eeg_features.SLOW_WAVE_FEATURE_NAMES:
                self.assertIn(
                    f'{channel_name}_{feature_name}',
                    eeg_processing.EEG_FEATURE_NAMES,
                )
        self.assertEqual(
            len(eeg_processing.EEG_FEATURE_NAMES),
            eeg_processing.EEG_FEATURE_LENGTH,
        )


if __name__ == '__main__':
    unittest.main()
