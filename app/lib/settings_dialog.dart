import 'package:flutter/material.dart';

import 'api_client.dart';
import 'app_theme.dart';

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
      backgroundColor: AppColors.surface,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      title: const Text('Servidor de predicción',
          style: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.w700)),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'IP y puerto de la PC que corre server/main.py '
            '(debe estar en la misma red WiFi que el celular).',
            style: TextStyle(fontSize: 13, color: AppColors.textSecondary),
          ),
          const SizedBox(height: 14),
          TextField(
            controller: _controller,
            keyboardType: TextInputType.url,
            style: const TextStyle(color: AppColors.textPrimary),
            decoration: InputDecoration(
              labelText: 'URL del servidor',
              labelStyle: const TextStyle(color: AppColors.textSecondary),
              hintText: 'http://192.168.1.100:8000',
              hintStyle: const TextStyle(color: AppColors.textMuted),
              filled: true,
              fillColor: AppColors.surfaceAlt,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: const BorderSide(color: AppColors.border),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: const BorderSide(color: AppColors.border),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: const BorderSide(color: AppColors.accent),
              ),
            ),
          ),
          const SizedBox(height: 12),
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
                    : const Icon(Icons.wifi_tethering_rounded, size: 18),
                label: const Text('Probar conexión'),
              ),
              if (_testOk == true)
                const Icon(Icons.check_circle_rounded, color: AppColors.accent2)
              else if (_testOk == false)
                const Icon(Icons.error_rounded, color: AppColors.danger),
            ],
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancelar', style: TextStyle(color: AppColors.textSecondary)),
        ),
        FilledButton(
          onPressed: () => Navigator.of(context).pop(_controller.text.trim()),
          style: FilledButton.styleFrom(backgroundColor: AppColors.accent),
          child: const Text('Guardar'),
        ),
      ],
    );
  }
}
