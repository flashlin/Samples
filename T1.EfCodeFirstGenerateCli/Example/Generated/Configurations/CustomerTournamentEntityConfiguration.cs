using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CustomerTournamentEntityConfiguration : IEntityTypeConfiguration<CustomerTournamentEntity>
    {
        public void Configure(EntityTypeBuilder<CustomerTournamentEntity> builder)
        {
            builder.ToTable("CustomerTournament");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Currency)
                .HasColumnType("varchar(3)")
                .IsRequired()
                .HasMaxLength(3)
            ;

            builder.Property(x => x.LoginName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.TournamentType)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.TournamentCategory)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.JoinDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.TotalWinLose)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.HistoryWinLose)
                .HasColumnType("decimal(19,6)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.TotalWinningBet)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.TotalLosingBet)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.TotalBet)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.HistoryWinningBet)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.HistoryLosingBet)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.HistoryTotalBet)
                .HasColumnType("int(10,0)")
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
