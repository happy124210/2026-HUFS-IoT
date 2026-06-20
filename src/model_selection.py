GATE_FALSE_ALARM_MAX = 0.05
GATE_EVENT_RECALL_MIN = 0.70


def passes_deployment_gate(candidate, baseline):
    """Decide promotion from validation metrics only."""
    return (
        candidate['macro_f1'] > baseline['macro_f1']
        and candidate['normal_false_alarm_rate'] <= GATE_FALSE_ALARM_MAX
        and candidate['recall']['glass'] >= GATE_EVENT_RECALL_MIN
        and candidate['recall']['scream'] >= GATE_EVENT_RECALL_MIN
    )


def rank_key(validation_metrics):
    return (
        validation_metrics['macro_f1'],
        -validation_metrics['normal_false_alarm_rate'],
        min(
            validation_metrics['recall']['glass'],
            validation_metrics['recall']['scream'],
        ),
    )
