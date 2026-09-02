import 'package:flutter/material.dart';

import 'api_client.dart';

/// Diálogo simple para configurar la URL del servidor (server/main.py),
/// ej: http://192.168.1.100:8000
class SettingsDialog extends StatefulWidget {
  final String currentUrl;

  const SettingsDialog({super.key, required this.currentUrl});

  @override
  State<SettingsDialog> createState() => _SettingsDialogState();
}

class _SettingsDialogState extends State<SettingsDialog> {
  late final TextEditingController _controller;
  bool _testing = false;
  bool? _testOk;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: widget.currentUrl);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _testConnection() async {
    setState(() {
      _testing = true;
      _testOk = null;
    });
    final ok = await ApiClient.checkHealth(_controller.text);
    if (!mounted) return;
    setState(() {
      _testing = false;
      _testOk = ok;
    });
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Servidor de predicción'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'IP y puerto de la PC que corre server/main.py '
            '(debe estar en la misma red WiFi que el celular).',
            style: TextStyle(fontSize: 13, color: Colors.black54),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _controller,
            keyboardType: TextInputType.url,
            decoration: const InputDecoration(
              labelText: 'URL del servidor',
              hintText: 'http://192.168.1.100:8000',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              TextButton.icon(
                onPressed: _testing ? null : _testConnection,
                icon: _testing
                    ? const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.wifi_tethering, size: 18),
                label: const Text('Probar conexión'),
              ),
              if (_testOk == true)
                const Icon(Icons.check_circle, color: Colors.green)
              else if (_testOk == false)
                const Icon(Icons.error, color: Colors.red),
            ],
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancelar'),
        ),
        FilledButton(
          onPressed: () => Navigator.of(context).pop(_controller.text.trim()),
          child: const Text('Guardar'),
        ),
      ],
    );
  }
}
