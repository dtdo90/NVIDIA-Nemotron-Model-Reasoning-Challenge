# Exact v5 Representative Traces

Source CSV: `data/single_phase_training_clean/single_phase_sft_v5.csv`

Each file below is copied from one row of `single_phase_sft_v5.csv`.
The prompt and assistant trace are verbatim from the CSV; only the metadata header is added for navigation.

| Problem type | Subtype/source_mode | Rows in subtype | Selected ID | Answer check | File |
|---|---|---:|---|---|---|
| Bit Manipulation | `huikang_real_bit` | 1364 | `b1f5a2e8` | matched | [bit_manipulation__huikang_real_bit__b1f5a2e8.txt](bit_manipulation__huikang_real_bit__b1f5a2e8.txt) |
| Bit Manipulation | `huikang_real_bit_extra_trace` | 238 | `50e57462` | matched | [bit_manipulation__huikang_real_bit_extra_trace__50e57462.txt](bit_manipulation__huikang_real_bit_extra_trace__50e57462.txt) |
| Bit Manipulation | `huikang_synthetic_matching` | 4515 | `huikang_matching_03c56cf6` | n/a_no_boxed_answer_format | [bit_manipulation__huikang_synthetic_matching__huikang_matching_03c56cf6.txt](bit_manipulation__huikang_synthetic_matching__huikang_matching_03c56cf6.txt) |
| Gravity | `real` | 1597 | `233e854d` | matched | [gravity__real__233e854d.txt](gravity__real__233e854d.txt) |
| Numeral System | `real` | 1576 | `0122d53a` | matched | [numeral_system__real__0122d53a.txt](numeral_system__real__0122d53a.txt) |
| Numeric Equation Transformation Rules | `real` | 651 | `047c4111` | matched | [numeric_equation_transformation_rules__real__047c4111.txt](numeric_equation_transformation_rules__real__047c4111.txt) |
| Numeric Equation Transformation Rules | `synthetic` | 3879 | `syn_ne_direct_template_extra_0134_0001` | matched | [numeric_equation_transformation_rules__synthetic__syn_ne_direct_template_extra_0134_0001.txt](numeric_equation_transformation_rules__synthetic__syn_ne_direct_template_extra_0134_0001.txt) |
| Symbol Transform | `op_ab_guess_0134_correct` | 13 | `258b796b` | matched | [symbol_transform__op_ab_guess_0134_correct__258b796b.txt](symbol_transform__op_ab_guess_0134_correct__258b796b.txt) |
| Symbol Transform | `op_ab_guess_0134_wrong` | 151 | `2d624cab` | matched | [symbol_transform__op_ab_guess_0134_wrong__2d624cab.txt](symbol_transform__op_ab_guess_0134_wrong__2d624cab.txt) |
| Symbol Transform | `phase1_synthetic_direct_template` | 216 | `st_phase1_direct_real_remap_9fc69c17_00` | matched | [symbol_transform__phase1_synthetic_direct_template__st_phase1_direct_real_remap_9fc69c17_00.txt](symbol_transform__phase1_synthetic_direct_template__st_phase1_direct_real_remap_9fc69c17_00.txt) |
| Symbol Transform | `real` | 59 | `9fc69c17` | matched | [symbol_transform__real__9fc69c17.txt](symbol_transform__real__9fc69c17.txt) |
| Symbol Transform | `single_phase_synthetic_direct_template` | 325 | `st_single_phase_direct_template_0134_0000` | matched | [symbol_transform__single_phase_synthetic_direct_template__st_single_phase_direct_template_0134_0000.txt](symbol_transform__single_phase_synthetic_direct_template__st_single_phase_direct_template_0134_0000.txt) |
| Symbol Transform | `symbol_transform_unreliable_pattern_guess` | 600 | `6c7f24b7` | matched | [symbol_transform__symbol_transform_unreliable_pattern_guess__6c7f24b7.txt](symbol_transform__symbol_transform_unreliable_pattern_guess__6c7f24b7.txt) |
| Text Cipher | `real` | 1576 | `6f90f7c4` | matched | [text_cipher__real__6f90f7c4.txt](text_cipher__real__6f90f7c4.txt) |
| Text Cipher | `single_phase_synthetic_text_cipher_confusion` | 200 | `tc_confusion_synthetic_0095` | matched | [text_cipher__single_phase_synthetic_text_cipher_confusion__tc_confusion_synthetic_0095.txt](text_cipher__single_phase_synthetic_text_cipher_confusion__tc_confusion_synthetic_0095.txt) |
| Text Cipher | `text_cipher_decision_point_curriculum` | 200 | `syn_tc_dp_v1_extra_phrase_copy_alignment_0182` | matched | [text_cipher__text_cipher_decision_point_curriculum__syn_tc_dp_v1_extra_phrase_copy_alignment_0182.txt](text_cipher__text_cipher_decision_point_curriculum__syn_tc_dp_v1_extra_phrase_copy_alignment_0182.txt) |
| Text Cipher | `text_cipher_enhance_long` | 100 | `syn_tc_enhance_long_0065` | matched | [text_cipher__text_cipher_enhance_long__syn_tc_enhance_long_0065.txt](text_cipher__text_cipher_enhance_long__syn_tc_enhance_long_0065.txt) |
| Text Cipher | `text_cipher_enhance_twin` | 100 | `syn_tc_enhance_twin_0023` | matched | [text_cipher__text_cipher_enhance_twin__syn_tc_enhance_twin_0023.txt](text_cipher__text_cipher_enhance_twin__syn_tc_enhance_twin_0023.txt) |
| Text Cipher | `text_cipher_no_candidate_recovery` | 50 | `syn_tc_case0_no_candidate_0038` | matched | [text_cipher__text_cipher_no_candidate_recovery__syn_tc_case0_no_candidate_0038.txt](text_cipher__text_cipher_no_candidate_recovery__syn_tc_case0_no_candidate_0038.txt) |
| Text Cipher | `text_cipher_reread_fail_to_pass` | 200 | `syn_tc_dp_phrase_alignment_0039` | matched | [text_cipher__text_cipher_reread_fail_to_pass__syn_tc_dp_phrase_alignment_0039.txt](text_cipher__text_cipher_reread_fail_to_pass__syn_tc_dp_phrase_alignment_0039.txt) |
| Text Cipher | `text_cipher_reread_pass_to_fail` | 200 | `syn_tc_dp_v2_extra_phrase_repeated_letter_failed_0032` | matched | [text_cipher__text_cipher_reread_pass_to_fail__syn_tc_dp_v2_extra_phrase_repeated_letter_failed_0032.txt](text_cipher__text_cipher_reread_pass_to_fail__syn_tc_dp_v2_extra_phrase_repeated_letter_failed_0032.txt) |
| Unit Conversion | `real` | 1594 | `778c5108` | matched | [unit_conversion__real__778c5108.txt](unit_conversion__real__778c5108.txt) |
