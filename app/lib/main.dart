import 'package:material_ui/material_ui.dart';

import 'replay/replay_data.dart';
import 'replay/replay_screen.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BrickOrSwish',
      theme: ThemeData(colorSchemeSeed: Colors.deepOrange),
      darkTheme: ThemeData(colorSchemeSeed: Colors.deepOrange, brightness: Brightness.dark),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('BrickOrSwish')),
      body: ListView(
        children: [
          const _SectionHeader('Replay a recorded session'),
          for (final c in replayCases)
            ListTile(
              leading: const Icon(Icons.replay),
              title: Text(c.title),
              subtitle: Text(c.subtitle),
              trailing: const Icon(Icons.chevron_right),
              onTap: () => Navigator.of(context).push(
                MaterialPageRoute<void>(builder: (_) => ReplayScreen(replayCase: c)),
              ),
            ),
          const _SectionHeader('Live'),
          const ListTile(
            enabled: false,
            leading: Icon(Icons.videocam_outlined),
            title: Text('Camera'),
            subtitle: Text('Coming next: camera preview and tap to mark the hoop'),
          ),
        ],
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  const _SectionHeader(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 4),
      child: Text(text, style: Theme.of(context).textTheme.titleSmall),
    );
  }
}
