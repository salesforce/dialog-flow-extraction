"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

domain = ["taxi", "restaurant", "hotel", "attraction", "train"]
test_domain = "hotel"
train_domains = domain.copy()
train_domains.remove(test_domain)