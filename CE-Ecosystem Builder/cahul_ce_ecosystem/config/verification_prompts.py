"""Prompts for Cahul and Circular Economy verification.

This module contains the prompt template used to verify whether entities
are based in Cahul and related to Circular Economy activities.
"""
from __future__ import annotations

VERIFICATION_PROMPT = r'''
Based ONLY on the extracted information below, determine:

1. Is the entity based in Cahul, Moldova?
   Check for: postal codes MD-xxxx (especially MD-3000, MD-3901, MD-3905, MD-3907, MD-3909), phone numbers +373, Cahul region references (Cahul, Congaz, Giurgiulești, Raionul Cahul), "Cahul" or "Moldova" in addresses.
   If found: confidence 0.6-1.0 (based on strength of evidence)
   If not found: confidence 0.0, is_cahul_based = false
   Provide evidence string with EXACT text found (or empty string if not found).

2. Is the entity related to Circular Economy?
   Check for Romanian keywords: economie circulară, reciclare, sustenabilitate, durabil, durabilitate, gestionare deșeuri, gestionarea deșeurilor, reutilizare, reparare, renovare, reprelucrare, valorificare, eficiența resurselor, reducerea deșeurilor, zero deșeuri, deșeuri zero, dezvoltare durabilă, design circular, eco-design, economie verde, producție sustenabilă, consum responsabil, colectare deșeuri, sortare, centru de reciclare, platformă de reciclare, consultanță, cercetare și dezvoltare, inovație, parteneriat, colaborare.
   Check for Russian keywords: циркулярная экономика, экономика замкнутого цикла, переработка, устойчивость, устойчивое развитие, управление отходами, повторное использование, ремонт, восстановление, регенерация, ресурсоэффективность, сокращение отходов, ноль отходов, предотвращение отходов, замкнутый цикл, жизненный цикл, круговое проектирование, эко-дизайн, зеленая экономика, устойчивое производство, ответственное потребление, биоразлагаемый, вторичное сырье, сбор отходов, сортировка, центр переработки, консалтинг, исследования и разработки, инновации, партнерство, сотрудничество.
   Check for English keywords: circular economy, recycling, sustainability, waste management, reuse, repair, remanufacturing, resource efficiency, zero waste.
   If found: confidence based on number and relevance of keywords
   If not found: confidence 0.0, is_ce_related = false
   Provide evidence string with EXACT keywords found (or empty string if not found).

CRITICAL: Only use information from the provided text. Do not make assumptions. If information is missing, set to false/0.0/empty string.
'''

__all__ = ["VERIFICATION_PROMPT"]
