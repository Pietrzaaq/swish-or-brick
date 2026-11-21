import 'package:material_ui/material_ui.dart';

import 'replay_data.dart';

class ReplayScreen extends StatefulWidget {
  const ReplayScreen({super.key, required this.replayCase});

  final ReplayCase replayCase;

  @override
  State<ReplayScreen> createState() => _ReplayScreenState();
}

class _ReplayScreenState extends State<ReplayScreen> {
  late final Future<ReplayResult> _result = runReplay(widget.replayCase);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(widget.replayCase.title)),
      body: FutureBuilder<ReplayResult>(
        future: _result,
        builder: (context, snap) {
          if (snap.hasError) {
            return _Message(icon: Icons.error_outline, text: 'Replay failed:\n${snap.error}');
          }
          if (!snap.hasData) {
            return const Center(child: CircularProgressIndicator());
          }
          final r = snap.data!;
          return ListView(
            padding: const EdgeInsets.symmetric(vertical: 8),
            children: [
              _Summary(result: r),
              for (final s in r.shots) _ShotTile(shot: s),
            ],
          );
        },
      ),
    );
  }
}

class _Summary extends StatelessWidget {
  const _Summary({required this.result});

  final ReplayResult result;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final r = result;
    final makes = r.count('make');
    final attempts = r.shots.length;
    final pct = attempts == 0 ? 0 : (100 * makes / attempts).round();
    final allAgree = r.verdicts > 0 && r.agreed == r.verdicts;
    return Card(
      margin: const EdgeInsets.fromLTRB(16, 8, 16, 8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('$makes / $attempts', style: text.displaySmall),
            Text('made shots ($pct%)', style: text.titleMedium),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                _CountChip(outcome: 'make', count: makes),
                _CountChip(outcome: 'miss', count: r.count('miss')),
                _CountChip(outcome: 'uncertain', count: r.count('uncertain')),
              ],
            ),
            if (r.verdicts > 0) ...[
              const SizedBox(height: 12),
              Row(
                children: [
                  Icon(allAgree ? Icons.verified : Icons.error_outline,
                      color: allAgree ? _colour('make') : _colour('miss')),
                  const SizedBox(width: 8),
                  Expanded(child: Text('Matches your verdict: ${r.agreed} of ${r.verdicts}')),
                ],
              ),
            ],
            const SizedBox(height: 12),
            Text(
              'Shot logic on this device: ${r.frames} frames (${r.durationS.toStringAsFixed(0)} s of video) '
              'in ${r.logicTime.inMilliseconds} ms',
              style: text.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}

class _CountChip extends StatelessWidget {
  const _CountChip({required this.outcome, required this.count});

  final String outcome;
  final int count;

  @override
  Widget build(BuildContext context) {
    return Chip(
      avatar: Icon(_icon(outcome), color: _colour(outcome), size: 18),
      label: Text('$count ${_label(outcome).toLowerCase()}'),
    );
  }
}

class _ShotTile extends StatelessWidget {
  const _ShotTile({required this.shot});

  final ReplayShot shot;

  @override
  Widget build(BuildContext context) {
    final colour = _colour(shot.outcome);
    final match = shot.matchesTruth;
    return ListTile(
      leading: CircleAvatar(
        backgroundColor: colour.withValues(alpha: 0.15),
        child: Icon(_icon(shot.outcome), color: colour),
      ),
      title: Text('Shot ${shot.shot} · ${_label(shot.outcome)}'),
      subtitle: Text('at ${shot.tStart.toStringAsFixed(1)} s · ${shot.reason.replaceAll('_', ' ')}'),
      trailing: match == null
          ? null
          : Tooltip(
              message: match ? 'Matches your verdict' : 'You said: ${shot.truth}',
              child: Icon(match ? Icons.check_circle_outline : Icons.cancel_outlined,
                  color: match ? _colour('make') : _colour('miss')),
            ),
    );
  }
}

class _Message extends StatelessWidget {
  const _Message({required this.icon, required this.text});

  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [Icon(icon, size: 48), const SizedBox(height: 12), Text(text, textAlign: TextAlign.center)],
        ),
      ),
    );
  }
}

String _label(String outcome) => switch (outcome) {
      'make' => 'Make',
      'miss' => 'Miss',
      _ => 'Uncertain',
    };

IconData _icon(String outcome) => switch (outcome) {
      'make' => Icons.check,
      'miss' => Icons.close,
      _ => Icons.help_outline,
    };

Color _colour(String outcome) => switch (outcome) {
      'make' => Colors.green.shade600,
      'miss' => Colors.red.shade600,
      _ => Colors.amber.shade700,
    };
