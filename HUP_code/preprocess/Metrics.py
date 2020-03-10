class Metrics():
    @classmethod
    def div(cls, a, b):
        if (b == 0):
            return 0
        a, b = float(a), float(b)
        return a / b

    @classmethod
    def get_metrics_recall_mrr(cls, N, recall, mrr, total):
        recallN = Metrics.div(recall, total)
        mrrN = Metrics.div(mrr, total)
        print "Metrics: N:%d, recall@N: %f, mrr@N: %f, recall_cnt:%d, rank:%f, total_cnt:%d" % (
        N, recallN, mrrN, recall, mrr, total)

    @classmethod
    def get_metrics_recall_mrr_NList(cls, NList, dictRecall, dictMrr, total):
        print "\n"
        for i in range(len(NList)):
            N = NList[i]
            recall = dictRecall[N]
            mrr = dictMrr[N]
            Metrics.get_metrics_recall_mrr(N, recall, mrr, total)

    @classmethod
    def eval_test_result_example(cls, simpred, tarid, recall, mrr, NList):
        for N in NList:
            pr = set()
            flag = 0
            for (simpop_p, score) in simpred:
                if len(pr) >= N:
                    break
                if int(tarid) == int(simpop_p):
                    flag = 1
                    break
                pr.add(simpop_p)
                # top N has
            rank = len(pr) + 1

            if flag == 1:
                recall[N] += 1
                mrr[N] += 1 / float(rank)

        pass

    @classmethod
    def init_recall_mrr(cls, NList):
        recall = {}
        mrr = {}
        for N in NList:
            recall[N] = 0
            mrr[N] = 0
        return [recall, mrr]

