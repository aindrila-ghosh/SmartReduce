import scipy.stats


def mcnemar_p(b, c):
  """Computes McNemar's test.
  Args:
    b: the number of "wins" for the first condition.
    c: the number of "wins" for the second condition.
  Returns:
    A p-value for McNemar's test.
  """
  n = b + c
  x = min(b, c)
  dist = scipy.stats.binom(n, .5)
  return 2. * dist.cdf(x)


def mcnemar_midp(b, c):
  """Computes McNemar's test using the "mid-p" variant.
  This is based closely on:
    
  M.W. Fagerland, S. Lydersen, P. Laake. 2013. The McNemar test for 
  binary matched-pairs data: Mid-p and asymptotic are better than exact 
  conditional. BMC Medical Research Methodology 13: 91.
  Args:
    b: the number of "wins" for the first condition.
    c: the number of "wins" for the second condition.
  Returns:
    A p-value for the mid-p variant of McNemar's test.
  """
  x = min(b, c)
  n = b + c
  dist = scipy.stats.binom(n, .5)
  return mcnemar_p(b, c) - dist.pmf(x)